
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.inference.predictor import Predictor
from src.backtesting.simulator import Simulator
from src.models.regime.hmm import RegimeDetector
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester")

# === EMBEDDED UTILS TO AVOID IMPORT ERRORS ===

class LazySequenceDataset(Dataset):
    """
    Zero-Copy Dataset that slices data on-the-fly.
    Copied from train.py to ensure standalone execution.
    """
    def __init__(self, features, time_features, targets, seq_len):
        # Store as Float32 Tensors (Shared Memory)
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        
        # Pre-process Targets
        # Target [0,1] -> Direction [-1, 1], Confidence [1]
        t_dir = np.where(targets > 0.5, 1.0, -1.0)
        t_conf = np.ones_like(t_dir)
        self.targets = torch.FloatTensor(np.stack([t_dir, t_conf], axis=1))
        
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        # Slice atomic window
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, t, y

# =============================================

def run_backtest():
    """
    1. Load Val Data (Parquet)
    2. Run Model Inference (ONNX)
    3. Generate Signals
    4. Run Simulation (VectorBT)
    """
    val_path = config.PROCESSED_DATA_DIR / "val.parquet"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        return

    logger.info(f"Loading Validation Data: {val_path}")
    df_val = pd.read_parquet(val_path)
    
    if 'close' not in df_val.columns:
        logger.error("Close price missing in validation data. Cannot backtest.")
        return

    # Prepare features for Model
    # Identify Feature Cols (columns starting with 'PC_' or others used in train)
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    feature_cols = [c for c in df_val.columns if c not in non_feat and 'sample_weight' not in c]
    
    # Prioritize PCA if present
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    if pca_cols:
        logger.info(f"Using PCA Columns: {pca_cols}")
        feature_cols = pca_cols
    else:
        logger.info(f"Using Raw Columns: {feature_cols}")
    
    # Time Features
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    # If missing, zero fill (shouldn't happen in val.parquet)
    for c in time_cols:
        if c not in df_val.columns: df_val[c] = 0.0
    
    # Arrays
    features = df_val[feature_cols].values.astype(np.float32)
    time_features = df_val[time_cols].values.astype(np.float32)
    targets = df_val['target'].values 
    
    # Handle NaNs in features (Zero Fill)
    features = np.nan_to_num(features)
    time_features = np.nan_to_num(time_features)

    seq_len = 10 # MUST MATCH TRAIN.PY DEFAULT (updated from 60 to 10 in train.py refactor)
    # BE CAREFUL: Check train.py seq_len. In the refactored main(), it was `seq_len = 10`.
    
    logger.info(f"Initializing Dataset with Seq Len: {seq_len}")
    
    # Dataset
    dataset = LazySequenceDataset(features, time_features, targets, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    # Predictor
    try:
        predictor = Predictor(model_name="agent1")
    except Exception as e:
        logger.error(f"Could not load model: {e}. Make sure agent1.onnx exists.")
        return

    # Load HMM for Filter
    hmm_path = config.MODELS_DIR / "hmm_regime.pkl"
    if hmm_path.exists():
        logger.info(f"Loading HMM from {hmm_path}...")
        hmm_model = joblib.load(hmm_path)
        # We need to predict states for the validation set.
        # HMM expects [log_ret, vol]. We need to reconstruct this or use RegimeDetector helper?
        # RegimeDetector.prepare_data needs a DF with 'close'.
        # We have df_val.
        
        # Helper wrapper
        rd = RegimeDetector(n_components=3)
        rd.model = hmm_model
        rd.is_fitted = True
        
        # Predict all states
        # catch warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # prepare_data returns a subset DataFrame
            # We need to align it with df_val
            # The HMM might have diff length due to rolling windows in prepare_data
            # prepare_data does dropna().
            
            # Let's do it manually on aligned data to represent "Real-Time" knowledge
            # Actually, prepare_data computes log_ret and rolling vol.
            # This causes NaNs at start.
            
            # Let's use the helper but be careful with alignment.
            # We only care about the valid_df part (iloc[seq_len:])
            
            # Calculate features on full df_val
            hmm_data = rd.prepare_data(df_val) 
            # Predict
            states = rd.model.predict(hmm_data.values)
            
            # Align: hmm_data index vs df_val index
            # Reindex states to match df_val
            s_series = pd.Series(states, index=hmm_data.index)
            df_val['regime'] = s_series.reindex(df_val.index).fillna(-1) # -1 for unknown
            
        logger.info("HMM States generated.")
    else:
        logger.warning("HMM Model not found. Running without Regime Filter.")
        df_val['regime'] = 1 # Default to allow everything
    
    all_probs = []
    
    logger.info("Running Batch Inference...")
    for batch_x, batch_t, _ in tqdm(loader):
        batch_x_np = batch_x.numpy()
        batch_t_np = batch_t.numpy()
        preds = predictor.predict(batch_x_np, batch_t_np)
        all_probs.extend(preds['direction'])
        
    all_probs = np.array(all_probs)
    
    # Align Signals with DF
    valid_df = df_val.iloc[seq_len:].copy()
    valid_df['pred_score'] = all_probs
    
    # Signal Logic
    # 1. Confidence Threshold (Stricter)
    THRESHOLD = 0.95
    
    # 2. Regime Filter
    # We want to AVOID State 0 (Low Vol/Range/Bearish determined by training logs)
    # State 0 Mean Ret was Negative in logs.
    # State 1 Mean Ret was Positive.
    # State 2 Mean Ret was High Positive (Volatile).
    # ALLOW: State 1 and 2. BLOCK: State 0.
    
    # Check if 'regime' column exists
    if 'regime' in valid_df.columns:
        # 0 is blocked
        regime_condition = (valid_df['regime'] != 0) 
    else:
        regime_condition = True

    # Combined Signal
    # Buy (1)
    buy_signal = (valid_df['pred_score'] > THRESHOLD) & regime_condition
    
    # Exit (-1)
    # Exit if Score drops below 0.0 OR Regime becomes 0 (Panic/Bear switch)?
    # Let's keep specific exit logic simple first
    sell_signal = (valid_df['pred_score'] < 0.0)
    
    conditions = [buy_signal, sell_signal]
    choices = [1, -1]
    valid_df['signal_trade'] = np.select(conditions, choices, default=0)
    
    logger.info(f"Generated {np.sum(valid_df['signal_trade'] == 1)} Entries and {np.sum(valid_df['signal_trade'] == -1)} Exits.")
    logger.info(f"Regime Filter skipped {len(valid_df[ (valid_df['pred_score'] > THRESHOLD) & (~regime_condition) ])} trades.")
    
    # Simulation
    sim = Simulator(fees=0.001, slippage=0.0005)
    portfolio, stats = sim.run_backtest(valid_df['close'], valid_df['signal_trade'])
    
    # Print Stats
    print("\n" + "="*40)
    print("      STRATEGY VALIDATION REPORT      ")
    print("="*40)
    print(stats[['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']])
    print("="*40 + "\n")

if __name__ == "__main__":
    run_backtest()
