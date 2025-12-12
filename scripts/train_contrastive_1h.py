import os
import resource
try:
    import psutil
except ImportError:
    psutil = None

# CRITICAL FIX: Prevent OpenMP Deadlock on RunPod
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import sys
import argparse
import logging
import requests
import torch
import traceback
import gc
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent
from src.training.contrastive_trainer import ContrastiveTrainer
from src.training.validation import AdversarialValidator
from src.data.datasets import MaskedTimeSeriesDataset

logger = logging.getLogger(__name__)

def log_memory(step=""):
    if psutil:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        logger.info(f"💾 MEMORY [{step}]: {mem:.1f} MB")
    else:
        logger.info(f"💾 MEMORY [{step}]: N/A (psutil missing)")

def terminate_pod():
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        logger.warning("Auto-Kill Switch: Skipped.")
        return
    logger.info(f"INITIATING SELF-DESTRUCTION for Pod {pod_id}...")
    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    query = f"""mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}"""
    requests.post(url, json={'query': query})

class LazySequenceDataset(torch.utils.data.Dataset):
    """
    Zero-Copy Dataset that slices data on-the-fly.
    """
    def __init__(self, features, time_features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        t_dir = np.where(targets > 0.5, 1.0, -1.0)
        t_conf = np.ones_like(t_dir)
        self.targets = torch.FloatTensor(np.stack([t_dir, t_conf], axis=1))
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, t, y

def main(args):
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting CONTRASTIVE LEARNING (1H) Experiment (Phase 30)...")
        log_memory("Start")
        
        # Override Config for Logic awareness (files are manual though)
        config.TIMEFRAME = '1h'
        
        # 1. Load Data (Isolated 1H)
        train_path = config.PROCESSED_DATA_DIR / 'train_1h.parquet'
        test_path = config.PROCESSED_DATA_DIR / 'val_1h.parquet'
        
        if not train_path.exists():
            raise FileNotFoundError(f"1H Data not found at {train_path}. Run pipeline_1h.py first!")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Features
        exclude = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        feature_cols = [c for c in train_df.columns if c not in exclude]
        pca_cols = [c for c in feature_cols if c.startswith('PC_')]
        if pca_cols: feature_cols = pca_cols
        
        feat_dim = len(feature_cols)
        seq_len = 10
        
        # Prepare Data
        def prepare_raw(df):
            feats = df[feature_cols].astype(np.float32).values
            time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            if all([c in df.columns for c in time_cols]):
                time_feats = df[time_cols].astype(np.float32).values
            else:
                time_feats = np.random.randn(len(df), 4).astype(np.float32)
            targets = df['target'].astype(np.float32).values
            return feats, time_feats, targets
            
        train_feats, train_times, train_targets = prepare_raw(train_df)
        val_feats, val_times, val_targets = prepare_raw(test_df)
        
        del train_df, test_df
        gc.collect()
        
        # Datasets
        dataset = LazySequenceDataset(train_feats, train_times, train_targets, seq_len)
        val_dataset = LazySequenceDataset(val_feats, val_times, val_targets, seq_len)
        
        BATCH_SIZE = 1024
        NUM_WORKERS = 8
        
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Model
        # Using Phase 27b Best Params
        model = TransformerAgent(input_dim=feat_dim, d_model=128, nhead=4, num_layers=2, dropout=0.12)
        
        # Trainer
        trainer = ContrastiveTrainer(model, lr=args.lr)
        
        # Step 1: SupCon Pre-Training
        logger.info("=== STEP 1: SupCon Feature Learning (1H) ===")
        model = trainer.train_contrastive(train_loader, epochs=args.supcon_epochs)
        
        # Step 2: Linear Probing
        logger.info("=== STEP 2: Linear Probing (Classification 1H) ===")
        trainer.train_linear_probe(train_loader, val_loader, epochs=args.probe_epochs)
        
        # Verify result
        final_loss = trainer.validate(val_loader)
        logger.info(f"✅ Final Validation Loss (Contrastive Agent 1H): {final_loss:.4f}")
        
        # Export (Isolated Name)
        out_path = config.MODELS_DIR / "model_contrastive_1h.pt"
        torch.save(model.state_dict(), out_path)
        logger.info(f"Saved model to {out_path}")
        
    except Exception as e:
        logger.error(f"Crash: {e}")
        traceback.print_exc()
    finally:
        if args.autokill: terminate_pod()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supcon_epochs", type=int, default=5)
    parser.add_argument("--probe_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--autokill", action="store_true")
    args = parser.parse_args()
    main(args)
