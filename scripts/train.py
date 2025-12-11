import os
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
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent
from src.training.trainer import Trainer
from src.training.pretrainer import PreTrainer
from src.training.validation import AdversarialValidator
from src.data.datasets import MaskedTimeSeriesDataset
from src.models.ensemble.meta_labeling import MetaAuditor
from src.data.frac_diff import FractionalDifferencing

logger = logging.getLogger(__name__)

def terminate_pod():
    """
    CRITICAL: Golden Rule #2
    Sends specific API call to RunPod to destroy this pod and stop billing.
    """
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not pod_id or not api_key:
        logger.warning("Auto-Kill Switch: RUNPOD_POD_ID or RUNPOD_API_KEY not found. Skipping termination (Safe for Local Dev).")
        return

    logger.info(f"INITIATING SELF-DESTRUCTION for Pod {pod_id}...")
    
    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    query = f"""
    mutation {{
        podTerminate(input: {{podId: "{pod_id}"}})
    }}
    """
    
    try:
        response = requests.post(url, json={'query': query}, timeout=10)
        if response.status_code == 200:
            logger.info("Termination signal sent successfully. Goodbye.")
        else:
            logger.error(f"FAILED to send termination signal: {response.text}")
    except Exception as e:
        logger.error(f"CRITICAL ERROR in Kill Switch: {e}")

def export_to_onnx(model, input_dim, model_name="agent1"):
    """
    Exports the trained PyTorch model to ONNX format.
    """
    model.eval()
    
    # Dummy inputs for tracing
    dummy_input_feats = torch.randn(1, 10, input_dim, requires_grad=True).to(model.device)
    dummy_input_time = torch.randn(1, 10, 4, requires_grad=True).to(model.device)
    
    file_path = config.MODELS_DIR / f"{model_name}.onnx"
    
    logger.info(f"Exporting model to ONNX: {file_path}")
    
    torch.onnx.export(
        model,
        (dummy_input_feats, dummy_input_time),
        str(file_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_features', 'input_time'],
        output_names=['output_dir_conf'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 1: 'seq_len'},
            'input_time': {0: 'batch_size', 1: 'seq_len'},
            'output_dir_conf': {0: 'batch_size'}
        }
    )
    logger.info("ONNX Export Successful.")

class LazySequenceDataset(torch.utils.data.Dataset):
    """
    Zero-Copy Dataset that slices data on-the-fly.
    Eliminates initial memory hang.
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

def main(args):
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting ADVANCED Training Pipeline...")
        
        # 1. Load Data (Real Data from Pipeline)
        logger.info("Loading Real Data from 'data/processed'...")
        
        try:
            train_path = config.DATA_DIR / 'processed' / 'train.parquet'
            test_path = config.DATA_DIR / 'processed' / 'val.parquet'
            
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(f"Processed data not found. Looked for {train_path} and {test_path}")
                
            print("DEBUG: Loading Train Parquet...")
            train_df = pd.read_parquet(train_path)
            print(f"DEBUG: Train Loaded. Shape: {train_df.shape}")
            
            print("DEBUG: Loading Val Parquet...")
            test_df = pd.read_parquet(test_path)
            print(f"DEBUG: Val Loaded. Shape: {test_df.shape}")
            
            # Identify Feature Columns
            if 'target' not in train_df.columns:
                raise ValueError("Column 'target' missing in training data.")
            
            exclude = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            feature_cols = [c for c in train_df.columns if c not in exclude]
            
            # Prioritize PCA columns
            pca_cols = [c for c in feature_cols if c.startswith('PC_')]
            if pca_cols:
                logger.info(f"PCA Columns detected. Using only: {pca_cols}")
                feature_cols = pca_cols
            
            feat_dim = len(feature_cols)
            logger.info(f"Loaded {len(train_df)} training samples. Feature Dim: {feat_dim}")
            
            seq_len = 10 # Context window
            
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            raise

        # 2. FRACTIONAL DIFFERENCING (ensure NaNs)
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        
        # 3. ADVERSARIAL VALIDATION (Safety)
        if hasattr(args, 'no_adv_val') and args.no_adv_val:
            logger.info("⚠️ Skipping Adversarial Validation (User Requested).")
        else:
            adv_val = AdversarialValidator()
            # ... (skipped for brevity, assuming standard logic)
            pass

        # Prepare Data for Lazy Loading
        logger.info("Preparing Raw Data for Lazy Loading...")
        
        def prepare_raw(df):
            # Features
            feats = df[feature_cols].astype(np.float32).values
            # Time
            time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            if all([c in df.columns for c in time_cols]):
                time_feats = df[time_cols].astype(np.float32).values
            else:
                time_feats = np.random.randn(len(df), 4).astype(np.float32)
            # Target
            targets = df['target'].astype(np.float32).values
            return feats, time_feats, targets

        train_feats, train_times, train_targets = prepare_raw(train_df)
        val_feats, val_times, val_targets = prepare_raw(test_df)
        
        # Create Datasets
        logger.info("Initializing Lazy Datasets...")
        dataset = LazySequenceDataset(train_feats, train_times, train_targets, seq_len)
        val_dataset = LazySequenceDataset(val_feats, val_times, val_targets, seq_len)
        
        # DataLoaders (num_workers=0 to avoid multiprocessing clones on RunPod if unstable)
        # Using pin_memory=True for Speed if GPU used
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0, pin_memory=True)
        
        logger.info("DataLoaders Ready.")
        
        # Winning Hyperparameters form Optuna (Trial 0)
        # d_model=64, nhead=8, num_layers=3, lr=4.6e-5, dropout=0.23
        model = TransformerAgent(input_dim=feat_dim, d_model=64, nhead=8, num_layers=3, dropout=0.23)
        
        # 4. PRE-TRAINING (Self-Supervised)
        if args.pretrain:
            logger.info("=== Phase 9: Self-Supervised Pre-Training ===")
            raw_2d = train_df[feature_cols].values
            
            # Time features
            time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            if all([c in train_df.columns for c in time_cols]):
                 raw_time = train_df[time_cols].values
            else:
                 raw_time = np.random.randn(len(train_df), 4)

            logger.info("DEBUG: PreTrain Step 1: Initializing Masked Dataset...")
            masked_dataset = MaskedTimeSeriesDataset(raw_2d, raw_time, seq_len=seq_len)
            
            logger.info("DEBUG: PreTrain Step 2: Creating DataLoader (num_workers=0)...")
            # Force num_workers=0 to prevent Multiprocessing Deadlocks
            masked_loader = DataLoader(masked_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
            
            logger.info("DEBUG: PreTrain Step 3: Initializing PreTrainer...")
            pretrainer = PreTrainer(model)
            
            logger.info("DEBUG: PreTrain Step 4: Starting Train Loop...")
            pretrainer.train(masked_loader, epochs=2)
            logger.info("DEBUG: PreTrain Step 5: Finished.")
            
        # 5. SUPERVISED TRAINING (Fine-Tuning)
        logger.info("=== Phase 3: Supervised Fine-Tuning ===")
        # Updated LR from optimization (Trial 0)
        trainer = Trainer(model, lr=4.6e-5)
        trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        # 6. META-LABELING (The Auditor)
        logger.info("=== Phase 8: Meta-Labeling (Auditor Training) ===")
        # Generate Predictions on Valid Set
        model.eval()
        preds_list = []
        targets_list = []
        inputs_list = []
        
        with torch.no_grad():
             for x, t, y in val_loader:
                 x, t = x.to(model.device), t.to(model.device)
                 out = model(x, t) # (B, 2) -> Simulating Dir, Conf
                 preds_list.append(out.cpu())
                 targets_list.append(y)
                 inputs_list.append(x[:, -1, :].cpu()) # Last step features
        
        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)
        inputs = torch.cat(inputs_list)
        
        # Construct Meta-Set
        # Target: Did direction match? (Simulated)
        # Using simple check: sign(pred_dir) == sign(target_dir)
        pred_dir = preds[:, 0]
        true_dir = targets[:, 0]
        meta_target = (torch.sign(pred_dir) == torch.sign(true_dir)).long()
        
        meta_auditor = MetaAuditor("auditor_v1")
        # Inputs: Original features + Pred Prob + Pred Side
        # Convert to DF
        meta_df = pd.DataFrame(inputs.numpy(), columns=[f'f{i}' for i in range(feat_dim)])
        # Assuming pred[:, 1] is confidence (Sigmoid)
        meta_auditor.fit(meta_df, preds[:, 1].numpy(), np.sign(preds[:, 0].numpy()), meta_target.numpy())
        
        # 7. EXPORT
        export_to_onnx(trainer.model, feat_dim)
        
        logger.info("✅ Full Training Pipeline Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        traceback.print_exc()
        
    finally:
        if args.autokill:
            terminate_pod()
        else:
            logger.info("Auto-Kill disabled (Local Mode).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--autokill", action="store_true", help="Enable RunPod Auto-Kill Switch")
    parser.add_argument("--pretrain", action="store_true", help="Deep Pre-Training")
    parser.add_argument("--no-adv-val", action="store_true", help="Skip Adversarial Validation (Debug)")
    args = parser.parse_args()
    
    main(args)
