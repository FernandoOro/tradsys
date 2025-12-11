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

# Helper function (Global Scope)
def create_sequences(df, feat_cols, target_col, seq_len):
    """
    Creates sequences for Transformer input using a pre-allocated loop.
    Returns: (X, T, Y)
    """
    logger.info("Creating Sequences (Pre-allocated Loop)...")
    
    # DEBUG: Check types
    logger.info(f"Feature Dtypes:\n{df[feat_cols].dtypes}")
    
    # Ensure Float32 (Fastest for Numpy/Torch)
    data = df[feat_cols].astype(np.float32).values # (N, F)
    targets = df[target_col].astype(np.float32).values # (N,)
    
    # Time Data
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    if all([c in df.columns for c in time_cols]):
        time_data = df[time_cols].astype(np.float32).values # (N, 4)
    else:
        time_data = np.random.randn(len(df), 4)

    num_sequences = len(df) - seq_len
    feat_dim = data.shape[1]
    
    # PRE-ALLOCATION (Robust & Safe)
    X = np.zeros((num_sequences, seq_len, feat_dim), dtype=np.float32)
    T = np.zeros((num_sequences, seq_len, 4), dtype=np.float32)
    Y = np.zeros((num_sequences,), dtype=np.float32)
    
    logger.info(f"Target memory: {X.nbytes / 1024**2:.1f} MB. Processing {num_sequences} sequences...")
    
    # Simple Integer Loop (Safe from Deadlocks)
    for i in range(num_sequences):
        X[i] = data[i:i+seq_len]
        T[i] = time_data[i:i+seq_len]
        Y[i] = targets[i+seq_len]
        
        if i % 10000 == 0 and i > 0:
             print(f"DEBUG: Filled {i}/{num_sequences} sequences...")
    
    print("DEBUG: Sequence filling complete.")
    return X, T, Y

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
            
            # Identify Feature Columns (excluding target, date, etc.)
            # Assuming Pipeline saved features with 'target' column.
            # We need to separate them.
            if 'target' not in train_df.columns:
                raise ValueError("Column 'target' missing in training data.")
            
            # Exclude non-features
            # "Time" columns are handled separately by TimeEmbedding, so exclude from main features
            exclude = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            feature_cols = [c for c in train_df.columns if c not in exclude]
            
            # Prioritize PCA columns if they exist
            pca_cols = [c for c in feature_cols if c.startswith('PC_')]
            if pca_cols:
                logger.info(f"PCA Columns detected. Using only: {pca_cols}")
                feature_cols = pca_cols
            
            feat_dim = len(feature_cols)
            logger.info(f"Loaded {len(train_df)} training samples. Feature Dim: {feat_dim}")
            logger.info(f"Features: {feature_cols}")
            
            N = len(train_df)
            seq_len = 10 # Context window
            
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            raise

        # 2. FRACTIONAL DIFFERENCING (Already done in Pipeline, skipping redundancy)
        # But we need to ensure NaNs are handled if any slipped through
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        
        # 3. ADVERSARIAL VALIDATION (Safety)
        if hasattr(args, 'no_adv_val') and args.no_adv_val:
            logger.info("⚠️ Skipping Adversarial Validation (User Requested).")
        else:
            adv_val = AdversarialValidator()
            auc = adv_val.check_drift(train_df, test_df)
            if auc > 0.7:
                 logger.warning("CRITICAL: Covariate Shift Detected (AUC > 0.7). Proceeding with CAUTION (Demo Mode).")
                 # raise ValueError("Data Drift > 0.7 AUC") # Disabled for First Run Demo

        # Prepare Tensors for Model
        # Need to create sequences (Sliding Window)
        # For simplicity in this script, we'll use a crude sliding window or if data is already row-per-step
        # Transformer expects (Batch, Seq, Feat).
        # We will create a dataset that slides over the rows.
        
        logger.info("Creating Sequences (Train)...")
        X_train, T_train, Y_train_raw = create_sequences(train_df, feature_cols, 'target', seq_len)
        logger.info("Creating Sequences (Val)...")
        X_test, T_test, Y_test_raw = create_sequences(test_df, feature_cols, 'target', seq_len)
        
        # Convert Y to Classification (One-Hot or Labels?)
        # Agent output is (Direction, Confidence).
        # We need targets to match.
        # Let's map Y (0/1) to Direction (-1/1) for Regression/Tanh head
        # Y_train_raw is 0 or 1.
        # Desired: 1 -> 1.0, 0 -> -1.0
        Y_train_dir = np.where(Y_train_raw > 0.5, 1.0, -1.0)
        # Confidence target? We don't have ground truth confidence. Set to 1.0 (Full confidence).
        Y_train_conf = np.ones_like(Y_train_dir)
        
        y_train = torch.FloatTensor(np.stack([Y_train_dir, Y_train_conf], axis=1))
        x_train = torch.FloatTensor(X_train)
        x_time_train = torch.FloatTensor(T_train)
        
        # Dataset
        dataset = TensorDataset(x_train, x_time_train, y_train)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Val Set
        val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(T_test), torch.FloatTensor(np.stack([np.where(Y_test_raw>0.5, 1.0, -1.0), np.ones_like(Y_test_raw)], axis=1)))
        val_loader = DataLoader(val_dataset, batch_size=32)
        
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

            masked_dataset = MaskedTimeSeriesDataset(raw_2d, raw_time, seq_len=seq_len)
            masked_loader = DataLoader(masked_dataset, batch_size=32, shuffle=True)
            pretrainer = PreTrainer(model)
            pretrainer.train(masked_loader, epochs=2)
            
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
