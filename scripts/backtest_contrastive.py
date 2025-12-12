
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

# === SIMULATOR (Local override for Contrastive SL/TP) ===
import vectorbt as vbt
class ContrastiveSimulator:
    """
    Isolated Simulator for Contrastive Strategy.
    Supports SL/TP execution explicitly.
    """
    def __init__(self, fees: float = 0.001, slippage: float = 0.0005):
        self.fees = fees
        self.slippage = slippage
        
    def run_backtest(self, close_price: pd.Series, open_price=None, high_price=None, low_price=None, signals=None, sl_stop=None, tp_stop=None):
        entries = signals == 1
        # If SL/TP is used, we disable 'exits' based on signal, 
        # unless signal is explicitly -1 (which we might keep for Safety Exit)
        exits = signals == -1 
        
        logger.info(f"Running Contrastive Simulation (SL={sl_stop}, TP={tp_stop})...")
        
        # Use OHLC for accurate SL/TP hit detection (Wicks)
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
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {stats['Total Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        
        return portfolio, stats

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
    
    hmm_states = None
    if hmm_path.exists() and pca_path.exists():
        logger.info(f"Loading Latent HMM from {hmm_path}...")
        hmm_model = joblib.load(hmm_path)
        pca_model = joblib.load(pca_path)
        
        # We need to extract embeddings check
        logger.info("Extracting Embeddings for Regime Detection...")
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
    else:
        logger.warning("Latent HMM not found. Running without filter.")

    # Inference
    all_probs = []
    logger.info("Running Batch Inference (PyTorch)...")
    for batch_x, batch_t in tqdm(loader, desc="Predictions"):
        preds = predictor.predict(batch_x.numpy(), batch_t.numpy())
        all_probs.extend(preds['direction'])
    all_probs = np.array(all_probs)
    
    # Alignment
    valid_df = df_val.iloc[seq_len:].copy()
    
    # SMOOTIIING: Fix Signal Flicker
    # Raw scores oscillate +/- 1 too fast. We Smooth it to capture the Trend.
    valid_df['pred_score_raw'] = all_probs
    valid_df['pred_score'] = pd.Series(all_probs).ewm(span=5).mean().values
    
    # DEBUG: Prediction Stats
    logger.info(f"Prediction Stats (Smoothed): Min={valid_df['pred_score'].min():.4f}, Max={valid_df['pred_score'].max():.4f}, Mean={valid_df['pred_score'].mean():.4f}, Std={valid_df['pred_score'].std():.4f}")
    
    # Assign HMM States
    if hmm_states is not None:
        valid_df['regime'] = hmm_states
        
        # --- PER REGIME STRATEGY ANALYSIS ---
        print("\n" + "="*65)
        print("       HMM LATENT REGIME PERFORMANCE")
        print("="*65)
        print(f"{'Regime':<8} | {'Bars':<6} | {'Mkt Ret':<10} | {'Win Rate':<10} | {'Avg Vol%':<10} | {'Strat PnL':<10}")
        print("-" * 65)
        
        # Calculate ATR globally first for analysis
        high_low = valid_df['high'] - valid_df['low']
        high_close = np.abs(valid_df['high'] - valid_df['close'].shift())
        low_close = np.abs(valid_df['low'] - valid_df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().bfill()
        valid_df['atr_pct'] = atr / valid_df['close']
        
        for r in sorted(valid_df['regime'].unique()):
            mask = valid_df['regime'] == r
            subset = valid_df[mask].copy()
            
            # 1. Market Context
            mkt_ret = subset['close'].pct_change().mean() * 10000 # bps
            avg_vol = subset['atr_pct'].mean() * 100
            
            # 2. Strategy Performance (Hypothetical)
            # Buy if score > THRESHOLD
            sigs = subset['pred_score'] > 0.99
            if sigs.sum() > 0:
                wins = ((subset['pred_score'] > 0.99) & (subset['target'] > 0.5)).sum()
                trades = sigs.sum()
                wr = (wins / trades) * 100
            else:
                wr = 0.0
                trades = 0
            
            print(f"{r:<8} | {mask.sum():<6} | {mkt_ret:<10.2f} | {wr:<10.1f}% | {avg_vol:<10.4f}% | {trades} trades")
            
        print("="*65 + "\n")
        
        # FILTER STRATEGY: 
        # We need High Accuracy AND High Volatility (>0.2%)
        # Let's verify which regime meets this.
        # For now, default to Regime 0 but user will see the table.
        logger.info("Applying Filter: KEEP ONLY REGIME 0")
        regime_condition = (valid_df['regime'] == 0)
        
    else:
        valid_df['regime'] = 1 
        regime_condition = True

    # Signal Logic
    THRESHOLD = 0.99
    buy_signal = (valid_df['pred_score'] > THRESHOLD) & regime_condition
    
    # Sell Logic:
    # DISABLE Score-based exit to prevent "flicker churn".
    # Only exit if we leave the Safe Regime (Safety Net).
    # Actual trade management is handled by SL/TP params in Simulator.
    sell_signal = (valid_df['regime'] != 0) 
    
    conditions = [buy_signal, sell_signal]
    choices = [1, -1]
    valid_df['signal_trade'] = np.select(conditions, choices, default=0)
    
    logger.info(f"Trades Generated (Entry Signals): {np.sum(valid_df['signal_trade'] == 1)}")
    
    # Sim
    # ALIGNMENT: Use ATR-based SL/TP (Same as TBM Training)
    # TBM Labeler used 1.5 * ATR.
    
    # Calculate ATR (True Range)
    high_low = valid_df['high'] - valid_df['low']
    high_close = np.abs(valid_df['high'] - valid_df['close'].shift())
    low_close = np.abs(valid_df['low'] - valid_df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().bfill()
    
    # Convert ATR to Percentage for VectorBT
    atr_pct = atr / valid_df['close']
    
    # Dynamic Stops
    dynamic_tp_pct = atr_pct * 1.5
    dynamic_sl_pct = atr_pct * 1.5
    
    # VOLATILITY FILTER:
    # DISABLE for "Lab Test" (Zero Fee Verification)
    # We want to see if the Model has Alpha in a vacuum.
    min_vol_threshold = 0.000 
    vol_filter = dynamic_tp_pct > min_vol_threshold
    
    logger.info(f"Applying Volatility Filter (Min TP > {min_vol_threshold*100:.2f}%)...")
    valid_df['signal_trade'] = valid_df['signal_trade'] & vol_filter
    
    logger.info(f"Avg ATR%: {atr_pct.mean()*100:.2f}% | Avg TP%: {dynamic_tp_pct.mean()*100:.2f}%")
    logger.info(f"Trades AFTER Vol Filter: {np.sum(valid_df['signal_trade'] == 1)}")

    # LAB TEST: Zero Fees, Zero Slippage, CLOSE-ONLY (Match Labeler Physics)
    # The Labeler used Close prices to determine Barrier Hits.
    # The OHLC Backtest gets stopped out by Wicks that the Labeler ignored.
    # To prove the Model works on its own terms, we must simulate "Close-Only" execution.
    sim = ContrastiveSimulator(fees=0.0, slippage=0.0)
    portfolio, stats = sim.run_backtest(
        close_price=valid_df['close'],
        open_price=None, # Disable OHLC
        high_price=None, # Disable OHLC
        low_price=None,  # Disable OHLC
        signals=valid_df['signal_trade'],
        sl_stop=dynamic_sl_pct,
        tp_stop=dynamic_tp_pct
    )
    
    print("\n" + "="*40)
    print("   CONTRASTIVE STRATEGY REPORT   ")
    print("="*40)
    print(stats[['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']])
    print("="*40 + "\n")

if __name__ == "__main__":
    run_backtest()
