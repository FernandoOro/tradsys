
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent
from src.backtesting.simulator import Simulator
from src.models.regime.hmm import RegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester-Contrastive")

# === PYTORCH PREDICTOR (Specific to this experiment) ===
class ContrastivePredictor:
    def __init__(self, model_path, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Architecture (Must match Phase 27b/Contrastive Config)
        # d_model=128, nhead=4, num_layers=2, dropout=0.12
        self.model = TransformerAgent(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.12)
        
        # Load Weights
        logger.info(f"Loading PyTorch Model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, features, time_features):
        """
        PyTorch Inference
        """
        x = torch.FloatTensor(features).to(self.device)
        t = torch.FloatTensor(time_features).to(self.device)
        
        with torch.no_grad():
            out = self.model(x, t) # (Batch, 2)
            
        return {
            "direction": out[:, 0].cpu().numpy(),
            "confidence": out[:, 1].cpu().numpy()
        }

# === DATASET (Copied for isolation) ===
class LazySequenceDataset(Dataset):
    def __init__(self, features, time_features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        # Dummy targets
        self.targets = torch.zeros(len(features), 2) 
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        # We don't need y for inference
        return x, t

def run_backtest():
    val_path = config.PROCESSED_DATA_DIR / "val.parquet"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        return

    logger.info(f"Loading Validation Data: {val_path}")
    df_val = pd.read_parquet(val_path)
    
    # Feature Selection (Same as Train)
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    feature_cols = [c for c in df_val.columns if c not in non_feat and 'sample_weight' not in c]
    
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    if pca_cols: feature_cols = pca_cols
    
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = df_val[feature_cols].values.astype(np.float32)
    time_features = df_val[time_cols].values.astype(np.float32)
    
    features = np.nan_to_num(features)
    time_features = np.nan_to_num(time_features) # Val data usually clean
    
    seq_len = 10 
    feat_dim = len(feature_cols)
    
    # Dataset
    dataset = LazySequenceDataset(features, time_features, df_val['target'].values, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    # Predictor
    model_path = config.MODELS_DIR / "model_contrastive.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
        
    predictor = ContrastivePredictor(model_path, input_dim=feat_dim)
    
    # Contrastive HMM Logic
    hmm_path = config.MODELS_DIR / "hmm_contrastive.pkl"
    pca_path = config.MODELS_DIR / "pca_contrastive.pkl"
    
    if hmm_path.exists() and pca_path.exists():
        logger.info("Loading Latent HMM & PCA...")
        hmm_model = joblib.load(hmm_path)
        pca_model = joblib.load(pca_path)
        
        # We need to extract embeddings for valid_df
        # Re-create dataset for inference just like before
        # But we need the embeddings this time, not just predictions
        # Predictor doesn't return embeddings. 
        # We need to access model definition. Predictor has `self.model`.
        
        # Let's extract embeddings for the whole validation set
        logger.info("Extracting Embeddings for Regime Detection...")
        
        # Re-use loader
        embeddings = []
        with torch.no_grad():
             for batch_x, batch_t in tqdm(loader, desc="Embeddings"):
                 batch_x = batch_x.to(predictor.device)
                 batch_t = batch_t.to(predictor.device)
                 emb = predictor.model.extract_features(batch_x, batch_t)
                 embeddings.append(emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        # PCA -> HMM
        latent_val = pca_model.transform(embeddings)
        hmm_states = hmm_model.predict(latent_val)
        
        # Analyze Regimes (Delayed until valid_df creation)
            
    else:
        logger.warning("Latent HMM not found. Running without filter.")
        hmm_states = None
    
    # Inference
    all_probs = []
    logger.info("Running Batch Inference (PyTorch)...")
    for batch_x, batch_t in tqdm(loader, desc="Predictions"):
        preds = predictor.predict(batch_x.numpy(), batch_t.numpy())
        all_probs.extend(preds['direction'])
    
    all_probs = np.array(all_probs)
    
    # Alignment
    valid_df = df_val.iloc[seq_len:].copy()
    valid_df['pred_score'] = all_probs
    
    # Assign HMM States if available
    if hmm_states is not None:
        valid_df['regime'] = hmm_states
        
        # Print Stats per Regime
        print("\n=== HMM REGIME ANALYSIS ===")
        for r in sorted(valid_df['regime'].unique()):
            mask = valid_df['regime'] == r
            r_ret = valid_df.loc[mask, 'close'].pct_change().mean() * 10000 # bps
            count = mask.sum()
            print(f"Regime {r}: {count} bars | Mean Return (bps): {r_ret:.4f}")
        print("===========================\n")
    else:
        valid_df['regime'] = 1 
    
    # Signal Logic
    THRESHOLD = 0.85
    
    # DYNAMIC REGIME FILTERING (Experimental)
    # Based on training log: State 2 had High Variance (Confusion).
    # We suspect State 2 is the bad one.
    regime_condition = True 
    if 'regime' in valid_df.columns and hmm_states is not None:
        # Assuming State 2 is the confused one (User can change this after seeing logs)
        # For now, let's NOT filter, just observe.
        pass

    buy_signal = (valid_df['pred_score'] > THRESHOLD) & regime_condition
    sell_signal = (valid_df['pred_score'] < 0.0)
    
    conditions = [buy_signal, sell_signal]
    choices = [1, -1]
    valid_df['signal_trade'] = np.select(conditions, choices, default=0)
    
    logger.info(f"Trades Generated: {np.sum(valid_df['signal_trade'] == 1)}")
    
    # Sim
    sim = Simulator(fees=0.001, slippage=0.0005)
    portfolio, stats = sim.run_backtest(valid_df['close'], valid_df['signal_trade'])
    
    print("\n" + "="*40)
    print("   CONTRASTIVE STRATEGY REPORT   ")
    print("="*40)
    print(stats[['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']])
    print("="*40 + "\n")

if __name__ == "__main__":
    run_backtest()
