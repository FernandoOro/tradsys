
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
import vectorbt as vbt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Validate-Futures")

# === PYTORCH PREDICTOR REUSED ===
class ContrastivePredictor:
    def __init__(self, model_path, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Consistent with 1H Architecture
        self.model = TransformerAgent(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.12)
        
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

# === DATASET REUSED ===
class LazySequenceDataset(Dataset):
    def __init__(self, features, time_features, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        return x, t

# === ECONOMICS SIMULATOR (FUTURES EDITION) ===
class FuturesSimulator:
    """
    Simulates Futures Trading with Funding Rates and Specific Fee Schedule.
    """
    def __init__(self, maker_fee=0.0002, taker_fee=0.0005, funding_rate_8h=0.0001):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_rate_8h = funding_rate_8h
        
    def run_backtest(self, close_price: pd.Series, signals, sl_stop=None, tp_stop=None):
        logger.info(f"Running FUTURES Simulation | Maker: {self.maker_fee*100}% | Taker: {self.taker_fee*100}% | Funding: {self.funding_rate_8h*100}%/8h")
        
        # Entries: 1 (Long), -1 (Short)
        entries = signals == 1
        short_entries = signals == -1
        
        # Exits are managed by SL/TP mostly in this simplified view, 
        # or Reversals (Long -> Short) which VectorBT handles automatically if short_entries used.
        
        # Funding Rate Simulation (Approximate)
        # We add 'fees' to every bar? No, VBT doesn't support funding rate natively easily.
        # We will approximate by increasing the 'fees' parameter slightly to amortize funding.
        # Funding = 0.01% per 8h = 0.03% per day.
        # If avg trade duration is 6h, impact is small.
        # We will assume Funding is paid ONCE per trade for simplicity (conservative).
        effective_fees = self.taker_fee + self.funding_rate_8h

        portfolio = vbt.Portfolio.from_signals(
            close_price,
            entries=entries,
            short_entries=short_entries, # NEW: Enable Shorts
            fees=effective_fees,
            slippage=0.0001, # Minimal slippage in testnet/sim
            init_cash=10000,
            freq='1h',
            sl_stop=sl_stop,
            tp_stop=tp_stop
        )
        
        stats = portfolio.stats()
        logger.info("Futures Results:")
        logger.info(f"Total Return: {stats['Total Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        logger.info(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        
        return portfolio, stats

def run_validation():
    # 1. Load Data
    val_path = config.PROCESSED_DATA_DIR / "val_1h.parquet" 
    # Fallback to general val if specific 1h not found (Phase 30 used contrastive pipeline)
    if not val_path.exists():
         logger.warning(f"{val_path} not found, falling back to val.parquet (RISK OF DIM MISMATCH)")
         val_path = config.PROCESSED_DATA_DIR / "val.parquet"
         
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        return

    logger.info(f"Loading Validation Data: {val_path}")
    df_val = pd.read_parquet(val_path)
    
    # Feature Engineering (Align with Phase 30)
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    # FIX: Do NOT exclude 'PC_' columns, as we are loading PCA-reduced data (val_1h.parquet)
    feature_cols = [c for c in df_val.columns if c not in non_feat and 'sample_weight' not in c]
    
    # If PCA was used in Phase 30 training, we need it here. 
    # Checking if df_val already has features or needs pipeline.
    # Assuming df_val is pre-processed from pipeline.
    
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = df_val[feature_cols].values.astype(np.float32)
    time_features = df_val[time_cols].values.astype(np.float32)
    features = np.nan_to_num(features)
    
    seq_len = 10 
    
    # Dataset
    dataset = LazySequenceDataset(features, time_features, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    
    # 2. Predictor
    # Use 1H Model if available (Phase 30)
    model_path = config.MODELS_DIR / "contrastive_1h" / "model_contrastive_1h.pt"
    if not model_path.exists():
        # Fallback to generic
        model_path = config.MODELS_DIR / "model_contrastive.pt"
        
    predictor = ContrastivePredictor(model_path, input_dim=len(feature_cols))
    
    # 3. Inference
    all_probs = []
    logger.info("Running Inference...")
    for batch_x, batch_t in tqdm(loader):
        preds = predictor.predict(batch_x.numpy(), batch_t.numpy())
        all_probs.extend(preds['direction']) # Score
        
    all_probs = np.array(all_probs)
    
    # 4. Strategy Logic
    valid_df = df_val.iloc[seq_len:].copy()
    valid_df['pred_score'] = pd.Series(all_probs).ewm(span=5).mean().values
    
    # HMM Filter (Optional/Simulated)
    # Loading HMM is complex here without duplicating code. 
    # We will assume Regime 1 (Golden) allows trades for this Economic Test.
    # Ideally we load 'hmm_contrastive_1h.pkl'.
    
    hmm_path = config.MODELS_DIR / "contrastive_1h" / "hmm_contrastive_1h.pkl"
    if hmm_path.exists():
         hmm = joblib.load(hmm_path)
         # Need PCA latent... simplified: Assume Regime Filter PASSES for high confidence
         # or we skip HMM for this pure economic test of the "Alpha Core".
         pass
    
    THRESHOLD = 0.99
    
    # SIGNAL GENERATION (Bi-Directional)
    # Model Output is Tanh (-1 to 1)
    
    mean_score = valid_df['pred_score'].mean()
    logger.info(f"Score Stats: Mean={mean_score:.4f} (Expect ~0.0 if balanced Tanh)")
    
    long_signal = valid_df['pred_score'] > THRESHOLD
    short_signal = valid_df['pred_score'] < -THRESHOLD
    
    conditions = [long_signal, short_signal]
    choices = [1, -1]
    
    # VectorBT Signals: 1=LongEntry, -1=ShortEntry
    valid_df['signal'] = np.select(conditions, choices, default=0)
    
    # Volatility Sizing (ATR)
    # We need ATR for stops.
    high = valid_df['high']
    low = valid_df['low']
    close = valid_df['close']
    tr = np.maximum(high - low, np.abs(high - close.shift(1)))
    atr = tr.rolling(14).mean().bfill()
    atr_pct = atr / close
    
    sl_stop = atr_pct * 1.5
    tp_stop = atr_pct * 1.5
    
    # 5. Run Validation
    sim = FuturesSimulator(maker_fee=0.0002, taker_fee=0.0005) # Binance VIP0 Futures
    
    portfolio, stats = sim.run_backtest(
        close_price=valid_df['close'],
        signals=valid_df['signal'],
        sl_stop=sl_stop,
        tp_stop=tp_stop
    )
    
    # Save Report
    report_path = "output/validation_futures_report.txt"
    with open(report_path, "w") as f:
        f.write(str(stats))
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
