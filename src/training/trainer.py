import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from src.config import config
from src.data.augmentation import GaussianNoise

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for classification to address class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

import wandb # New

class Trainer:
    """
    Standard Training Loop for PyTorch Models.
    Updated for Phase 13: Sample Weights & Noise Injection.
    WandB Integration Added.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Augmentation
        self.noise = GaussianNoise(std=0.01)
        
        # Losses (Reduction=None to apply weights manually)
        self.criterion_cls = FocalLoss(reduction='none') 
        self.criterion_reg = nn.MSELoss(reduction='none')

        # WANDB Init
        if config.WANDB_API_KEY:
            wandb.login(key=config.WANDB_API_KEY)
            wandb.init(project="smart-spot-trader", config={"lr": lr})
            
        # Scheduler (Dynamic Correction)
        # Reduces LR if Val Loss doesn't improve for 5 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, trial=None):
        import optuna # Local import to avoid circular dependency
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            count = 0
            
            # Loader is expected to yield (x, time, y) OR (x, time, y, weights)
            # We need to handle flexible batch unpacking or ensure loader provides weights
            
            for batch in train_loader:
                # Flexible Unpacking
                if len(batch) == 4:
                    batch_x, batch_time, batch_y, batch_w = batch
                    batch_w = batch_w.to(self.device)
                else:
                    batch_x, batch_time, batch_y = batch
                    batch_w = None # No weights
                
                batch_x = batch_x.to(self.device)
                batch_time = batch_time.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Apply Noise Augmentation
                batch_x = self.noise(batch_x)
                
                self.optimizer.zero_grad()
                output = self.model(batch_x, batch_time)
                
                # Split output [Direction, Confidence]
                pred_dir = output[:, 0]
                target_dir = batch_y[:, 0]
                
                # Compute Loss per sample
                loss_per_sample = self.criterion_reg(pred_dir, target_dir)
                
                # Apply Sample Weights if dominant
                if batch_w is not None:
                    loss_weighted = loss_per_sample * batch_w
                    loss = loss_weighted.mean()
                else:
                    loss = loss_per_sample.mean()
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                count += 1
            
            # Validation
            val_loss = self.validate(val_loader)
            
            avg_train_loss = train_loss / count if count > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if wandb.run:
                wandb.log({"train_loss": avg_train_loss, "val_loss": val_loss, "epoch": epoch, "lr": self.optimizer.param_groups[0]['lr']})
            
            # Dynamic Correction
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.MODELS_DIR / "agent1_best.pth")
                
            # OPTUNA PRUNING CHECK
            if trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                # Validation usually doesn't use weights for scoring, or does to measure weighted error?
                # Usually standard error on validation.
                if len(batch) == 4:
                     batch_x, batch_time, batch_y, _ = batch
                else:
                     batch_x, batch_time, batch_y = batch
                
                batch_x = batch_x.to(self.device)
                batch_time = batch_time.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x, batch_time)
                # Validation metric: Simple Mean S.E.
                loss = nn.MSELoss()(output[:, 0], batch_y[:, 0])
                total_loss += loss.item()
                count += 1
        return total_loss / count if count > 0 else float('inf')
