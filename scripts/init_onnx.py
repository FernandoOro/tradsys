import sys
import os
import torch
import logging
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.models.agents.transformer import TransformerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_init():
    logger.info("Initializing Untrained Agent for Cold Start...")
    
    # 1. Load PCA to determine input dimension
    try:
        with open(config.MODELS_DIR / "pca.pkl", "rb") as f:
            pca = pickle.load(f)
        feat_dim = pca.pca.n_components_
        logger.info(f"Loaded PCA. Input Dimension: {feat_dim}")
    except FileNotFoundError:
        logger.error("PCA not found. Run pipeline first.")
        return

    # 2. Initialize Model (Untrained)
    model = TransformerAgent(input_dim=feat_dim, d_model=64)
    model.eval()
    
    # 3. Export to ONNX
    file_path = config.MODELS_DIR / "agent1.onnx"
    
    # Dummy Inputs [Batch, Seq, Dim]
    dummy_input_feats = torch.randn(1, 10, feat_dim)
    dummy_input_time = torch.randn(1, 10, 4)
    
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
    logger.info(f"✅ Exported cold-start model to {file_path}")

if __name__ == "__main__":
    export_init()
