
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent
from src.backtesting.simulator import Simulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester-Contrastive-1H")

# === PYTORCH PREDICTOR ===
class ContrastivePredictor:
    def __init__(self, model_path, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Architecture (Must match Phase 27b/Contrastive Config)
        self.model = TransformerAgent(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.12)
        
        # Load Weights
        logger.info(f"Loading PyTorch Model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, features, time_features):
        x = torch.FloatTensor(features).to(self.device)
        t = torch.FloatTensor(time_features).to(self.device)
        
        with torch.no_grad():
            out = self.model(x, t) # (Batch, 2)
            
        return {
            "direction": out[:, 0].cpu().numpy(),
            "confidence": out[:, 1].cpu().numpy()
        }
    
    def extract_embeddings(self, features, time_features):
        x = torch.FloatTensor(features).to(self.device)
        t = torch.FloatTensor(time_features).to(self.device)
        with torch.no_grad():
            return self.model.extract_features(x, t).cpu().numpy()

# === DATASET ===
class LazySequenceDataset(Dataset):
    def __init__(self, features, time_features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        self.targets = torch.zeros(len(features), 2) 
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        return x, t

# === SIMULATOR (Local override for Contrastive SL/TP) ===
import vectorbt as vbt
class ContrastiveSimulator:
    def __init__(self, fees: float = 0.001, slippage: float = 0.0005):
        self.fees = fees
        self.slippage = slippage
        
    def run_backtest(self, close_price: pd.Series, open_price=None, high_price=None, low_price=None, signals=None, sl_stop=None, tp_stop=None):
        entries = signals == 1
        exits = signals == -1 
        
        logger.info(f"Running Contrastive Simulation (SL={sl_stop}, TP={tp_stop})...")
        
        portfolio = vbt.Portfolio.from_signals(
            close_price,
            entries,
            exits,
            high=high_price,
            low=low_price,
            open=open_price,
            fees=self.fees,
            slippage=self.slippage,
            init_cash=10000,
            freq='1h',
            sl_stop=sl_stop,
            tp_stop=tp_stop
        )
        stats = portfolio.stats()
        return portfolio, stats

def run_backtest():
    # Override Config for Logic awareness
    config.TIMEFRAME = '1h'

    # Load 1H Data
    val_path = config.PROCESSED_DATA_DIR / "val_1h.parquet"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        return

    logger.info(f"Loading Validation Data: {val_path}")
    df_val = pd.read_parquet(val_path)
    
    # Feature Selection
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    feature_cols = [c for c in df_val.columns if c not in non_feat and 'sample_weight' not in c]
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    if pca_cols: feature_cols = pca_cols
    
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = df_val[feature_cols].values.astype(np.float32)
    time_features = df_val[time_cols].values.astype(np.float32)
    features = np.nan_to_num(features)
    time_features = np.nan_to_num(time_features)
    
    seq_len = 10 
    feat_dim = len(feature_cols)
    
    dataset = LazySequenceDataset(features, time_features, df_val['target'].values, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    # Predictor
    model_path = config.MODELS_DIR / "model_contrastive_1h.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
        
    predictor = ContrastivePredictor(model_path, input_dim=feat_dim)
    
    # --- REGIME DETECTION (On The Fly Training if needed) ---
    hmm_path = config.MODELS_DIR / "hmm_contrastive_1h.pkl" 
    pca_path = config.MODELS_DIR / "pca_contrastive_1h_latent.pkl"
    
    # Extract Embeddings
    logger.info("Extracting Embeddings for Regime Detection...")
    embeddings = []
    for batch_x, batch_t in tqdm(loader, desc="Embeddings"):
        emb = predictor.extract_embeddings(batch_x.numpy(), batch_t.numpy())
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Train or Load HMM
    if not hmm_path.exists():
        logger.info("Training HMM on 1H Embeddings (First Run)...")
        # PCA Reduce for HMM
        pca_latent = PCA(n_components=10, random_state=42)
        latent_features = pca_latent.fit_transform(embeddings)
        joblib.dump(pca_latent, pca_path)
        
        # Train HMM
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        hmm.fit(latent_features)
        joblib.dump(hmm, hmm_path)
    else:
        logger.info("Loading HMM...")
        pca_latent = joblib.load(pca_path)
        hmm = joblib.load(hmm_path)
        latent_features = pca_latent.transform(embeddings)

    hmm_states = hmm.predict(latent_features)

    # Inference Loops for Scores
    all_probs = []
    logger.info("Running Batch Inference...")
    for batch_x, batch_t in tqdm(loader, desc="Predictions"):
        preds = predictor.predict(batch_x.numpy(), batch_t.numpy())
        all_probs.extend(preds['direction'])
    all_probs = np.array(all_probs)
    
    # DataFrame Alignment
    valid_df = df_val.iloc[seq_len:].copy()
    valid_df['pred_score'] = pd.Series(all_probs).ewm(span=5).mean().values
    valid_df['regime'] = hmm_states
    
    # Reporting
    print("       HMM PERFORMANCE (1H)")
    print("-" * 65)
    print(f"{'Regime':<8} | {'Bars':<6} | {'Mkt Ret':<10} | {'Win Rate':<10} | {'Avg Vol%':<10}")
    
    # Calculate ATR
    high_low = valid_df['high'] - valid_df['low']
    high_close = np.abs(valid_df['high'] - valid_df['close'].shift())
    low_close = np.abs(valid_df['low'] - valid_df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().bfill()
    valid_df['atr_pct'] = atr / valid_df['close']
    
    best_regime = 0
    best_wr = -1.0
    
    for r in sorted(valid_df['regime'].unique()):
        mask = valid_df['regime'] == r
        subset = valid_df[mask].copy()
        mkt_ret = subset['close'].pct_change().mean() * 10000 
        avg_vol = subset['atr_pct'].mean() * 100
        sigs = subset['pred_score'] > 0.99
        
        trade_count = sigs.sum()
        if trade_count > 0:
            wins = ((subset['pred_score'] > 0.99) & (subset['target'] > 0.5)).sum()
            wr = (wins / trade_count) * 100
        else:
            wr = 0.0
            
        print(f"{r:<8} | {mask.sum():<6} | {mkt_ret:<10.2f} | {wr:<10.1f}% | {avg_vol:<10.4f}% | {trade_count} trades")
        
        # Selection Logic: Valid trade count and highest WR
        if trade_count > 10 and wr > best_wr:
            best_wr = wr
            best_regime = r
        
    logger.info(f"Applying Filter: KEEP ONLY REGIME {best_regime} (Best WR: {best_wr:.1f}%)")
    
    buy_signal = (valid_df['pred_score'] > 0.99) & (valid_df['regime'] == best_regime)
    sell_signal = (valid_df['regime'] != best_regime)
    
    conditions = [buy_signal, sell_signal]
    choices = [1, -1]
    valid_df['signal_trade'] = np.select(conditions, choices, default=0)
    
    # Dynamic SL/TP
    # 1H ATR is typically 1%. 1.5x ATR = 1.5%.
    # Fees are 0.2%. Margin = 1.3%. huge.
    atr_pct = valid_df['atr_pct']
    dynamic_tp_pct = atr_pct * 1.5
    dynamic_sl_pct = atr_pct * 1.5
    
    # Run Sim (REAL WORLD SETTINGS)
    # Fees = 0.1% per side (0.2% round trip)
    # Slippage = 0.05%
    sim = ContrastiveSimulator(fees=0.001, slippage=0.0005)
    
    portfolio, stats = sim.run_backtest(
        close_price=valid_df['close'],
        open_price=valid_df['open'],
        high_price=valid_df['high'],
        low_price=valid_df['low'],
        signals=valid_df['signal_trade'],
        sl_stop=dynamic_sl_pct,
        tp_stop=dynamic_tp_pct
    )
    
    print("\n" + "="*40)
    print("   CONTRASTIVE 1H RESULTS (REAL WORLD)   ")
    print("="*40)
    print(stats[['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']])
    print("="*40 + "\n")

if __name__ == "__main__":
    run_backtest()
