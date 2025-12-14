
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
    print(portfolio.stats(['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades']))
    print("="*50 + "\n")
    
if __name__ == "__main__":
    run_backtest()
