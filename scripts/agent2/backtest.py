
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import pickle
import vectorbt as vbt

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.agent2.model import ResNetAgent
from src.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(path):
    df = pd.read_parquet(path)
    # Metadata columns to exclude from features
    metadata = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion', 'target_class', 'target_ret']
    
    # Extract features for model
    feature_cols = [c for c in df.columns if c not in metadata]
    X = df[feature_cols].values
    
    # Return features and the full dataframe (for simulation)
    return torch.tensor(X, dtype=torch.float32), df

def run_backtest():
    logger.info("🚀 Starting Agent 2 Backtest (RunPod Validation)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    val_path = config.PROCESSED_DATA_DIR / "val_reversion.parquet"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        return

    logger.info("Loading Data...")
    X_val, df_val = load_processed_data(val_path)
    
    # 2. Load Model
    model_path = config.MODELS_DIR / "agent2_best.pt"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Did you run train_final.py?")
        return
        
    logger.info("Loading Model...")
    input_dim = X_val.shape[1]
    model = ResNetAgent(input_dim=input_dim, num_blocks=4, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Inference
    logger.info("Running Inference...")
    batch_size = 4096
    loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)
    
    all_preds = []
    
    with torch.no_grad():
        for bx in loader:
            bx = bx[0].to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    df_val['prediction'] = all_preds
    
    # --- DEBUG: DIAGNOSE SILENCE ---
    print("\n[DIAGNOSTIC] Label Distribution (Ground Truth):")
    print(df_val['target_reversion'].value_counts())
    print("\n[DIAGNOSTIC] Prediction Distribution (Model Output):")
    print(df_val['prediction'].value_counts())
    
    # Check Probability Distribution for Minority Classes
    buy_probs = []
    sell_probs = []
    with torch.no_grad():
        for bx in loader:
            bx = bx[0].to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            buy_probs.extend(probs[:, 1].cpu().numpy())
            sell_probs.extend(probs[:, 2].cpu().numpy())
            
    buy_probs = np.array(buy_probs)
    sell_probs = np.array(sell_probs)
    
    print(f"\n[DIAGNOSTIC] Buy Probabilities (Class 1):")
    print(f"  Max: {buy_probs.max():.4f} | Mean: {buy_probs.mean():.4f} | >0.2: {(buy_probs > 0.2).sum()}")
    print(f"[DIAGNOSTIC] Sell Probabilities (Class 2):")
    print(f"  Max: {sell_probs.max():.4f} | Mean: {sell_probs.mean():.4f} | >0.2: {(sell_probs > 0.2).sum()}")
    print("-" * 30)
    
    # --- ACTIVATION: THRESHOLD LOGIC ---
    # Argmax chooses class 0 because prob(0) is usually ~0.6 and prob(1) ~0.3.
    # But 0.3 is HUGE for a rare event. We activate if prob > Threshold.
    THRESHOLD = 0.25
    
    # 0 = Wait, 1 = Buy, 2 = Sell
    # Priority: If both > Threshold, take the higher one.
    
    final_preds = np.zeros(len(df_val), dtype=int)
    
    # Vectorized Logic
    # Mask where Buy > Threshold
    buy_mask = buy_probs > THRESHOLD
    # Mask where Sell > Threshold
    sell_mask = sell_probs > THRESHOLD
    
    # Conflict resolution: If both are high, pick max
    conflict_mask = buy_mask & sell_mask
    
    # Case 1: Only Buy
    final_preds[buy_mask & (~sell_mask)] = 1
    # Case 2: Only Sell
    final_preds[sell_mask & (~buy_mask)] = 2
    # Case 3: Conflict - compare values
    final_preds[conflict_mask] = np.where(
        buy_probs[conflict_mask] > sell_probs[conflict_mask], 1, 2
    )
    
    df_val['prediction'] = final_preds
    logger.info(f"Applying Threshold {THRESHOLD} -> Trades Activated: {(final_preds!=0).sum()}")
    
    # 4. Simulation Logic
    # Agent 2 Logic:
    # 1 = Buy Signal (Reversion UP)
    # 2 = Sell Signal (Reversion DOWN)
    # 0 = Wait
    
    entries = df_val['prediction'] == 1
    short_entries = df_val['prediction'] == 2 # If futures supported
    
    # Exits: 
    # For simplicity in this backtest, let's use Fixed Time or Reverse Signal + SL/TP
    # But implementation plan says "Use SL/TP from Triple Barrier Logic".
    # Here allows VectorBT to manage SL/TP.
    
    # TP = 1.0%, SL = 0.5% (As defined in Labeling)
    tp_stop = 0.01
    sl_stop = 0.005
    
    logger.info(f"Simulating Trades (TP={tp_stop*100}%, SL={sl_stop*100}%)...")
    
    portfolio = vbt.Portfolio.from_signals(
        df_val['close'],
        entries,
        short_entries=short_entries, # Dual direction!
        fees=0.001, # 0.1% per side
        slippage=0.0005, # 0.05% slippage
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        init_cash=10000,
        freq='1h'
    )
    
    # 5. Report
    print("\n" + "="*50)
    print("   AGENT 2 PERFORMANCE (VALIDATION SET)   ")
    print("="*50)
    # Call stats() without arguments to get the full default report
    # This avoids KeyError if specific metric names differ by version
    stats = portfolio.stats() 
    print(stats)
    print("="*50 + "\n")
    
if __name__ == "__main__":
    run_backtest()
