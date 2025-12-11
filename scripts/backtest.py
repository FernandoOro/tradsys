
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
    # Buy (1) if Score > 0.5
    # Close (-1) if Score < 0.0 
    conditions = [
        valid_df['pred_score'] > 0.5,
        valid_df['pred_score'] < 0.0
    ]
    choices = [1, -1]
    valid_df['signal_trade'] = np.select(conditions, choices, default=0)
    
    logger.info(f"Generated {np.sum(valid_df['signal_trade'] == 1)} Entries and {np.sum(valid_df['signal_trade'] == -1)} Exits.")
    
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
