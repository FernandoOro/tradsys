
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
    
    # TP = 0.5%, SL = 0.5% (As defined in NEW labeling)
    tp_stop = 0.005
    sl_stop = 0.005
    
    # --- OPTIMIZATION: SENSITIVITY ANALYSIS ---
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"\n{'='*20} SENSITIVITY ANALYSIS {'='*20}")
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return [%]':<10} | {'Sharpe':<10}")
    print("-" * 70)

    for THRESHOLD in thresholds:
        final_preds = np.zeros(len(df_val), dtype=int)
        
        # Vectorized Logic
        buy_mask = buy_probs > THRESHOLD
        sell_mask = sell_probs > THRESHOLD
        conflict_mask = buy_mask & sell_mask
        
        final_preds[buy_mask & (~sell_mask)] = 1
        final_preds[sell_mask & (~buy_mask)] = 2
        final_preds[conflict_mask] = np.where(buy_probs[conflict_mask] > sell_probs[conflict_mask], 1, 2)
        
        # Signals
        entries = pd.Series(final_preds == 1, index=df_val.index)
        short_entries = pd.Series(final_preds == 2, index=df_val.index)
        
        # Exits (24h Timeout)
        exits = entries.shift(24).fillna(False)
        short_exits = short_entries.shift(24).fillna(False)

        if entries.sum() + short_entries.sum() == 0:
            print(f"{THRESHOLD:<10} | {0:<8} | {'NaN':<10} | {0.0:<10} | {'NaN':<10}")
            continue

        # Simulation
        try:
             portfolio = vbt.Portfolio.from_signals(
                df_val['close'],
                entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                fees=0.001,
                slippage=0.0005,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                init_cash=10000,
                freq='1h'
            )
             
             stats = portfolio.stats()
             trades = stats['Total Trades']
             win_rate = stats['Win Rate [%]']
             ret = stats['Total Return [%]']
             sharpe = stats['Sharpe Ratio']
             
             print(f"{THRESHOLD:<10} | {trades:<8} | {win_rate:<10.2f} | {ret:<10.2f} | {sharpe:<10.2f}")
             
        except Exception as e:
            logger.error(f"Error at threshold {THRESHOLD}: {e}")

    print("="*70 + "\n")
    
if __name__ == "__main__":
    run_backtest()
