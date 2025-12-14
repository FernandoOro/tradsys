
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import optuna
import wandb

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Strict Isolation Imports
from src.agent2.model import ResNetAgent
from src.agent2.losses import FocalLoss, LabelSmoothingLoss
from src.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
USE_WANDB = True # Set to True for RunPod
N_TRIALS = 50     # Optuna Trials

def load_data_labeled(path):
    df = pd.read_parquet(path)
    
    # Metadata to exclude from Features
    metadata = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion', 'target_class', 'target_ret']
    
    # Input Features (X)
    X = df.drop(columns=[c for c in df.columns if c in metadata], errors='ignore').select_dtypes(include=[np.number]).values
    
    # Target Labels (Y) - Use 'target_reversion' (0, 1, 2)
    # 0 = Neutral, 1 = Buy, 2 = Sell
    if 'target_reversion' not in df.columns:
        raise ValueError(f"Column 'target_reversion' not found in {path}. Did you run Agent 2 Pipeline?")
        
    y = df['target_reversion'].values.astype(int)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def objective(trial):
    # 1. Hyperparameters to Tune
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    gamma = trial.suggest_float('focal_gamma', 1.0, 3.0) # For Focal Loss
    
    # Init WandB for this trial
    if USE_WANDB:
        wandb.init(
            project="agent2_reversion",
            config=trial.params,
            reinit=True,
            group="optuna_tuning"
        )
    
    # 2. Setup Data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_path = config.PROCESSED_DATA_DIR / "train_reversion.parquet"
    val_path = config.PROCESSED_DATA_DIR / "val_reversion.parquet"
    
    X_train, y_train = load_data_labeled(train_path)
    X_val, y_val = load_data_labeled(val_path)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    
    # 3. Model Setup
    input_dim = X_train.shape[1]
    model = ResNetAgent(input_dim=input_dim, num_blocks=4).to(device)
    
    # Initialize Weights from DAE (Transfer Learning) if available
    dae_path = config.MODELS_DIR / "dae_reversion.pt"
    if dae_path.exists():
        # Load DAE state
        dae_state = torch.load(dae_path, map_location=device)
        # Filter only encoder keys (encoder_proj, encoder_blocks)
        # ResNetAgent keys: input_proj, blocks...
        # ResNetDAE keys: encoder_proj, encoder_blocks...
        # We need to map them.
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        # Map encoder_proj -> input_proj
        # Map encoder_blocks -> blocks
        for k, v in dae_state.items():
            if k.startswith("encoder_proj"):
                new_k = k.replace("encoder_proj", "input_proj")
                pretrained_dict[new_k] = v
            elif k.startswith("encoder_blocks"):
                new_k = k.replace("encoder_blocks", "blocks")
                pretrained_dict[new_k] = v
            elif k.startswith("bn_enc"):
                new_k = k.replace("bn_enc", "bn_in")
                pretrained_dict[new_k] = v
                
        # Update current model
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # logger.info("Loaded Pre-trained DAE Weights!") # Silence for Optuna
    
    # 4. Loss & Optimizer
    # 3 Classes: 0 (Neutral), 1 (Buy), 2 (Sell)
    # Use Focal Loss to handle imbalance (Neutral is dominant)
    criterion = FocalLoss(gamma=gamma, reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 5. Training Loop
    EPOCHS = 10 
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx) 
            
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += by.size(0)
            correct += (predicted == by).sum().item()
            
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_val += by.size(0)
                correct_val += (predicted == by).sum().item()
                
        val_acc = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # Reporting
        trial.report(val_acc, epoch)
        
        if USE_WANDB:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch
            })
            
        if trial.should_prune():
            if USE_WANDB: wandb.finish()
            raise optuna.exceptions.TrialPruned()
            
    if USE_WANDB: wandb.finish()
    
    return val_acc

if __name__ == "__main__":
    if USE_WANDB:
        wandb.login(key=config.WANDB_API_KEY)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    logger.info(f"Best Trial: {study.best_trial.params}")
    logger.info(f"Best Val Acc: {study.best_value}")
    
    # Save Best Params
    # We could rerun training with best params here... but for now, just print.
