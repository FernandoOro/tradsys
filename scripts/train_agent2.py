
import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.resnet import ResNetAgent, ResNetAutoEncoder
from src.models.regime.hmm import RegimeDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainAgent2")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def export_agent2_onnx(model, input_dim, seq_len=10):
    model.eval()
    dummy_input = torch.randn(1, seq_len * input_dim, requires_grad=True).to(next(model.parameters()).device)
    
    # Organize artifacts in subfolder
    output_dir = config.MODELS_DIR / "agent2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / "agent2.onnx"
    logger.info(f"Exporting Agent 2 to: {file_path}")
    
    torch.onnx.export(
        model,
        dummy_input, # ResNet takes Flattened input
        str(file_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info("Agent 2 ONNX Export Success.")

def prepare_data(regime_filter=0):
    """
    Loads Data, predicts Regimes (or loads them), filters for Regime 0.
    Returns: X (Features), y (Targets)
    """
    logger.info("Loading Data...")
    train_path = config.PROCESSED_DATA_DIR / 'train.parquet'
    val_path = config.PROCESSED_DATA_DIR / 'val.parquet'
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Load HMM
    hmm_path = config.MODELS_DIR / "hmm_regime.pkl"
    if not hmm_path.exists():
        logger.error("HMM Model not found! Cannot filter regimes.")
        sys.exit(1)
        
    logger.info("Loading HMM for Regime Filtering...")
    rd = RegimeDetector(n_components=3)
    rd.load_model()
    
    # Identify Features for HMM (Must match HMM training)
    # HMM uses 'close' (pct_change internally) or similar.
    # rd.predict_state expects a DataFrame with 'close' (or whatever prepare_data needs).
    # train_df should have 'close'.
    
    logger.info("Predicting Regimes...")
    # We predict on the FULL dataset to respect rolling windows correctly?
    # Or just passing the DF is enough (RegimeDetector handles it).
    train_regimes = rd.predict_state(train_df)
    val_regimes = rd.predict_state(val_df)
    
    train_df['regime'] = train_regimes
    val_df['regime'] = val_regimes
    
    # FILTER: Keep only Regime 0 (Range/Sideways)
    logger.info(f"Filtering for Regime {regime_filter}...")
    train_subset = train_df[train_df['regime'] == regime_filter].copy()
    val_subset = val_df[val_df['regime'] == regime_filter].copy()
    
    logger.info(f"Train Subset: {len(train_subset)} samples (Total: {len(train_df)})")
    logger.info(f"Val Subset: {len(val_subset)} samples (Total: {len(val_df)})")
    
    if len(train_subset) < 100:
        logger.error("Not enough Regime 0 samples to train! Aborting.")
        sys.exit(1)
        
    # Prepare Features (Same as Agent 1? Yes, reuse PCA)
    exclude = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'regime', 'regime_prob', 'log_ret']
    
    feature_cols = [c for c in train_subset.columns if c not in exclude and 'date' not in c]
    # Prefer PCA
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    if pca_cols:
        feature_cols = pca_cols
        
    logger.info(f"Using {len(feature_cols)} Features: {feature_cols}")
    
    # Create Sequences locally or just use Instantaneous?
    # ResNetAgent is MLP-based.
    # Let's use a small window (Seq Len 10) flattened, similar to Agent 1 context.
    seq_len = 10
    
    def to_sequences(df):
        # We cannot use rolling on a subsets because rows are non-contiguous!
        # CRITICAL INTUITION: You cannot slide a window over filtered rows.
        # You must slide window over the ORIGINAL data, THEN filter by the Regime of the LAST candle.
        pass
        
    # Re-logic:
    # 1. Create Sequences from FULL DF.
    # 2. Assign Regime to each Sequence (based on Time T).
    # 3. Filter Sequences where Regime == 0.
    
    # To save RAM, let's do this efficiently.
    # We iterate the full DF, creating samples, check regime, append if 0.
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    def extract_filtered_samples(df, regimes):
        # df and regimes must be aligned
        xs, ys = [], []
        data_vals = df[feature_cols].values
        target_vals = df['target'].values
        regime_vals = regimes.values # Series
        
        for i in range(seq_len, len(df)):
            curr_regime = regime_vals[i]
            if curr_regime == regime_filter:
                # This sequence ends in a Regime 0 candle. Relevant for training.
                seq = data_vals[i-seq_len : i]
                # Flatten immediately? ResNet takes flattened (Seq*Dim)
                xs.append(seq.flatten())
                ys.append(target_vals[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    logger.info("Extracting Sequences (Slow Step)...")
    X_train, y_train = extract_filtered_samples(train_df, train_df['regime'])
    X_val, y_val = extract_filtered_samples(val_df, val_df['regime'])
    
    return X_train, y_train, X_val, y_val, len(feature_cols)

def train(args):
    device = get_device()
    logger.info(f"Using Device: {device}")
    
    X_train, y_train, X_val, y_val, feat_dim = prepare_data()
    
    # Create DataLoaders
    # Input Dim for ResNet = Seq_Len * Feat_Dim
    input_dim = X_train.shape[1] 
    
    batch_size = 2048 # High batch size for MLP
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size)
    
    # Model
    model = ResNetAgent(input_dim=input_dim, hidden_dim=args.hidden_dim, num_blocks=args.blocks).to(device)
    logger.info(f"Model Initialized: ResNet {args.blocks} Blocks, Input {input_dim}")
    
    # --- PRE-TRAINING (AutoEncoder) ---
    if args.pretrain:
        logger.info("⚡ Starting DAE Pre-training (Denoising Autoencoder)...")
        ae = ResNetAutoEncoder(input_dim=input_dim, hidden_dim=args.hidden_dim, num_blocks=args.blocks).to(device)
        ae_opt = optim.AdamW(ae.parameters(), lr=1e-3)
        ae_crit = nn.MSELoss()
        
        for epoch in range(20): # 20 Epochs enough for DAE usually
            ae.train()
            train_loss = 0
            for X, _ in train_loader: # Ignore labels
                X = X.to(device)
                ae_opt.zero_grad()
                out = ae(X)
                # Input reconstruction (compare to clean X)
                # (forward adds noise internally)
                loss = ae_crit(out, X) 
                loss.backward()
                ae_opt.step()
                train_loss += loss.item()
            if (epoch+1) % 5 == 0:
                logger.info(f"DAE Epoch {epoch+1}/20 | Recon Loss: {train_loss/len(train_loader):.6f}")
        
        # Transfer Weights to Agent
        logger.info("Transferring Pre-trained Encoder weights to Agent...")
        # Dictionary Copy
        agent_dict = model.state_dict()
        ae_dict = ae.state_dict()
        
        # Filter matching keys (Encoder parts)
        # ResNetAgent keys: input_proj.weight, blocks.0...
        # ResNetAE keys: encoder_proj.weight, encoder_blocks.0...
        # Need manual mapping or stricter naming.
        # Simple fix: Iterate and copy if shape matches and name corresponds.
        
        # Manual Copy for safety
        model.input_proj.weight.data = ae.encoder_proj.weight.data.clone()
        model.input_proj.bias.data = ae.encoder_proj.bias.data.clone()
        model.bn_in.weight.data = ae.bn_enc.weight.data.clone()
        model.bn_in.bias.data = ae.bn_enc.bias.data.clone()
        
        for i in range(args.blocks):
            model.blocks[i].load_state_dict(ae.encoder_blocks[i].state_dict())
            
        logger.info("Weights Transferred. Encoder is primed.")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Use Binary Cross Entropy for Direction (Target 0/1)
    
    # Basic Training Loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X).squeeze() # (B)
            # y is 0/1. BCE expects float.
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X).squeeze()
                loss = criterion(out, y)
                val_loss += loss.item()
                
                # Acc
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2%}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            # Save Checkpoint? Not implemented for brevity, save export at end.
            
    # Export
    export_agent2_onnx(model, feat_dim, seq_len=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--pretrain", action="store_true", help="Enable DAE Pre-training")
    
    args = parser.parse_args()
    train(args)
