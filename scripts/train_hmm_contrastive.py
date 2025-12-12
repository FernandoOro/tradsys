
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainLatentHMM")

# === DATASET UTIL (Same as others) ===
class LazySequenceDataset(Dataset):
    def __init__(self, features, time_features, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        return x, t

def train_latent_hmm():
    logger.info("Starting Latent Regime Detection Training...")
    
    # 1. Load Data
    data_path = config.PROCESSED_DATA_DIR / "train.parquet"
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        return
        
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Feature Config
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    feature_cols = [c for c in df.columns if c not in non_feat and 'sample_weight' not in c]
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    if pca_cols: feature_cols = pca_cols
    
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = df[feature_cols].values.astype(np.float32)
    time_features = df[time_cols].values.astype(np.float32)
    features = np.nan_to_num(features)
    time_features = np.nan_to_num(time_features)
    
    seq_len = 10
    
    # 2. Load Contrastive Model
    model_path = config.MODELS_DIR / "model_contrastive.pt"
    if not model_path.exists():
        logger.error(f"Contrastive Model not found at {model_path}")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(feature_cols)
    model = TransformerAgent(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.12)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Extract Embeddings
    logger.info("Extracting Latent Embeddings (Geometry)...")
    dataset = LazySequenceDataset(features, time_features, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    embeddings = []
    
    with torch.no_grad():
        for batch_x, batch_t in tqdm(loader):
            batch_x = batch_x.to(device)
            batch_t = batch_t.to(device)
            # Use extract_features
            emb = model.extract_features(batch_x, batch_t)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    logger.info(f"Embeddings Shape: {embeddings.shape}")
    
    # 4. Dimensionality Reduction (PCA)
    # HMM behaves badly in high dimensions (128). We reduce to 3 dominant geometric components.
    logger.info("Fitting PCA (128 -> 3)...")
    pca = PCA(n_components=3)
    latent_data = pca.fit_transform(embeddings)
    
    # Save PCA
    joblib.dump(pca, config.MODELS_DIR / "pca_contrastive.pkl")
    logger.info(f"PCA Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # 5. Train HMM
    logger.info("Fitting Gaussian HMM on Latent Geometry...")
    # n_components=3: likely (1) Clear Uptrend, (2) Clear Downtrend, (3) Confused/Noise
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    hmm.fit(latent_data)
    
    # Save HMM
    joblib.dump(hmm, config.MODELS_DIR / "hmm_contrastive.pkl")
    
    # 6. Analyze States
    # We can't know which state is which without plotting returns, but we can look at "variance".
    # High variance in embedding space usually means "Confused Model".
    for i in range(hmm.n_components):
        # Determinant of covariance matrix ~ Volume of the confidence ellipsoid
        cov_det = np.linalg.det(hmm.covars_[i])
        logger.info(f"State {i} Covariance Det (Confusion): {cov_det:.2e}")
        
    logger.info("✅ Latent HMM Trained and Saved.")

if __name__ == "__main__":
    train_latent_hmm()
