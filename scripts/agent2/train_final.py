
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import wandb

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.agent2.model import ResNetAgent
from src.agent2.losses import FocalLoss
from src.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- BEST PARAMS FROM RUNPOD (Trial 3) ---
BEST_PARAMS = {
    'lr': 2.5554092418003832e-05,
    'batch_size': 256,
    'dropout': 0.1299310081487177,
    'focal_gamma': 2.770672235352367
}

def load_data_labeled(path):
    df = pd.read_parquet(path)
    metadata = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion', 'target_class', 'target_ret']
    X = df.drop(columns=[c for c in df.columns if c in metadata], errors='ignore').select_dtypes(include=[np.number]).values
    y = df['target_reversion'].values.astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_final():
    logger.info("🏆 Starting Final Training with Winner Hyperparameters...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    train_path = config.PROCESSED_DATA_DIR / "train_reversion.parquet"
    val_path = config.PROCESSED_DATA_DIR / "val_reversion.parquet"
    X_train, y_train = load_data_labeled(train_path)
    X_val, y_val = load_data_labeled(val_path)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BEST_PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BEST_PARAMS['batch_size'])
    
    # Init Model
    model = ResNetAgent(input_dim=X_train.shape[1], num_blocks=4, num_classes=3).to(device)
    
    # Transfer Learning (Load DAE if exists)
    dae_path = config.MODELS_DIR / "dae_reversion.pt"
    if dae_path.exists():
        logger.info("Loading DAE Pre-trained Weights...")
        dae_state = torch.load(dae_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dae_state.items():
            if k.startswith("encoder"):
                new_k = k.replace("encoder_proj", "input_proj").replace("encoder_blocks", "blocks").replace("bn_enc", "bn_in")
                if new_k in model_dict:
                     pretrained_dict[new_k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # Optimizer & Loss
    criterion = FocalLoss(gamma=BEST_PARAMS['focal_gamma'], reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=BEST_PARAMS['lr'], weight_decay=1e-4)
    
    # Train
    EPOCHS = 20 # Train a bit longer for final model
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                _, predicted = torch.max(logits, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()
        
        acc = correct / total
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.5f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.MODELS_DIR / "agent2_best.pt")
            logger.info("💾 New Best Model Saved!")

    logger.info(f"✅ Final Training Done. Saved to {config.MODELS_DIR / 'agent2_best.pt'}")

if __name__ == "__main__":
    train_final()
