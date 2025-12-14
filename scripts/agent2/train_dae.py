
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.agent2.model import ResNetAutoEncoder
from src.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path):
    df = pd.read_parquet(path)
    # Exclude targets and non-numeric for training DAE
    # We want to reconstruct FEATURES.
    # In pipeline, we saved [preserved_cols, pca_features].
    # Which columns are the inputs? 
    # Usually we use Normalized Features BEFORE PCA? Or PCA Features?
    # DAE on PCA features is common for denoising.
    # Let's assume input features are numeric columns excluding targets.
    
    exclude = ['target', 'target_reversion', 'target_class', 'target_ret', 'open', 'high', 'low', 'close', 'volume', 'time']
    # Also exclude sin/cos time features if we don't want to reconstruct "Time" (trivial)
    # But reconstructing time helps learn seasonality. Let's keep them if present.
    
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("close_")]
    
    # Actually, pipeline saved PCA features as columns 0, 1, 2... or names?
    # PCAFeatureReducer usually returns numeric indices or strings.
    # Let's check pipeline logic. It concats d_preserved + d_pca.
    # d_pca comes from pd.DataFrame(d_pca).
    # so cols are likely 0, 1, 2...
    
    # Safe bet: Select all numeric columns that are NOT known metadata.
    metadata = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion', 'target_class', 'target_ret']
    
    X = df.drop(columns=[c for c in df.columns if c in metadata], errors='ignore').select_dtypes(include=[np.number]).values
    return torch.tensor(X, dtype=torch.float32)

def train_dae(epochs=50, batch_size=256, lr=1e-3, device='cuda'):
    logger.info("🚀 Starting DAE Pre-training (The 'Market Physics' Class)")
    
    train_path = config.PROCESSED_DATA_DIR / "train_reversion.parquet"
    val_path = config.PROCESSED_DATA_DIR / "val_reversion.parquet"
    
    X_train = load_data(train_path)
    X_val = load_data(val_path)
    
    logger.info(f"Training Data Shape: {X_train.shape}")
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)
    
    input_dim = X_train.shape[1]
    
    model = ResNetAutoEncoder(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            # Forward (Noise is added inside model)
            recon = model(x)
            loss = criterion(recon, x) # Compare against clean x
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), config.MODELS_DIR / "dae_reversion.pt")
            
    logger.info(f"✅ DAE Training Complete. Best Loss: {best_loss:.6f}")
    logger.info(f"Saved to {config.MODELS_DIR / 'dae_reversion.pt'}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")
    try:
        train_dae(device=device)
    except Exception as e:
        logger.error(f"Failed: {e}")
