import optuna
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.models.agents.transformer import TransformerAgent
from src.training.trainer import Trainer
from src.model_selection.cv import PurgedKFold

logger = logging.getLogger(__name__)

class Tuner:
    """
    Bayesian Hyperparameter Optimizer using Optuna.
    Updated for Phase 15: Purged K-Fold Cross Validation.
    """
    
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        
    def objective(self, trial):
        """
        Optuna Objective Function.
        """
        # 1. Sample Hyperparameters
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead_options = [h for h in [2, 4, 8] if d_model % h == 0]
        if not nhead_options: nhead_options = [1]
        nhead = trial.suggest_categorical("nhead", nhead_options)
        
        num_layers = trial.suggest_int("num_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        # 2. Setup Data (Real Data)
        train_path = config.DATA_DIR / 'processed' / 'train.parquet'
        train_df = pd.read_parquet(train_path)
        
        # Determine Features (exclude target/time)
        exclude = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        feature_cols = [c for c in train_df.columns if c not in exclude]
        # Prioritize PCA
        pca_cols = [c for c in feature_cols if c.startswith('PC_')]
        if pca_cols: feature_cols = pca_cols
            
        feat_dim = len(feature_cols)
        
        # Prepare Tensors (Simple Sliding Window for Tuner)
        def create_sequences(df, feat_cols, target_col, seq_len=10):
            data = df[feat_cols].values
            targets = df[target_col].values
            
            # Time features
            time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            if all([c in df.columns for c in time_cols]):
                 t_data = df[time_cols].values
            else:
                 t_data = np.zeros((len(df), 4))

            xs, ts, ys = [], [], []
            for i in range(len(df) - seq_len):
                xs.append(data[i:i+seq_len])
                ts.append(t_data[i:i+seq_len])
                ys.append(targets[i+seq_len])
            return np.array(xs), np.array(ts), np.array(ys)

        X, T, Y_raw = create_sequences(train_df, feature_cols, 'target')
        
        # Tensors
        x_data = torch.FloatTensor(X)
        x_time = torch.FloatTensor(T)
        
        # Targets: Direction (-1/1) and Confidence (1)
        # Y_raw is 0 or 1
        y_dir = np.where(Y_raw > 0.5, 1.0, -1.0)
        y_conf = np.ones_like(y_dir)
        y_data = torch.FloatTensor(np.stack([y_dir, y_conf], axis=1))
        
        # Convert to numpy for KFold
        # (PurgedKFold expects indices)
        
        # 3. Purged Interaction
        # We need t1 (event end times) for PurgedCV. 
        # Assuming timestamps are sequential integers for this index
        N = len(x_data)
        t1 = pd.Series(np.arange(N) + 1, index=np.arange(N)) 
        
        pkf = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.01)
        
        scores = []
        
        try:
             # CV Loop
             for fold_idx, (train_idx, val_idx) in enumerate(pkf.split(x_data.numpy())):
                 # Slice Data
                 xt_fold = x_data[train_idx]
                 tt_fold = x_time[train_idx]
                 yt_fold = y_data[train_idx]
                 
                 xv_fold = x_data[val_idx]
                 tv_fold = x_time[val_idx]
                 yv_fold = y_data[val_idx]
                 
                 # Loaders
                 train_ds = TensorDataset(xt_fold, tt_fold, yt_fold)
                 val_ds = TensorDataset(xv_fold, tv_fold, yv_fold)
                 
                 train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                 val_loader = DataLoader(val_ds, batch_size=32)
                 
                 # Model Init
                 model = TransformerAgent(input_dim=feat_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
                 trainer = Trainer(model, lr=lr)
                 
                 # Train (Short)
                 trainer.train(train_loader, val_loader, epochs=2)
                 
                 # Score (Validation Loss)
                 val_loss = trainer.validate(val_loader)
                 scores.append(val_loss)
                 
             avg_loss = np.mean(scores)
             return avg_loss
             
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')

    def run_optimization(self):
        study = optuna.create_study(direction="minimize")
        logger.info("Starting Optimization Study with PURGED CV...")
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tuner = Tuner(n_trials=5)
    tuner.run_optimization()
