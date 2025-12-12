
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import config

def train_ensemble(n_models=5, start_index=1):
    """
    Trains N models with different seeds and saves them as agent1_v{i}.onnx
    """
    print(f"🚀 Starting Bagging Ensemble Training ({n_models} members)...")
    
    ensemble_dir = config.MODELS_DIR / "ensemble"
    ensemble_dir.mkdir(exist_ok=True, parents=True)
    
    # Base command (delegates to main train.py)
    # Uses 'Champion' Params (Trial 31) because they are now defaults in train.py!
    base_cmd = ["python", "scripts/train.py", "--epochs", "30", "--no-adv-val"] 
    # Reduced epochs slightly (30) for ensemble members to save time, or keep 50?
    # Let's do 30. Better to have diverse somewhat-underfit models than identical overfit ones.
    
    for i in range(start_index, start_index + n_models):
        seed = 42 + i * 111 # Different seed per model (42+111=153, 264...)
        # Note: train.py doesn't have a --seed arg yet, we might need to add it or rely on random initialization?
        # PyTorch initializes randomly by default unless seeded. 
        # Our train.py DOES NOT set a global seed (checked logic).
        # So running it multiple times yields different weights. Perfect.
        
        print(f"\n==========================================")
        print(f"🤖 Training Member {i}/{n_models}...")
        print(f"==========================================")
        
        try:
            subprocess.run(base_cmd, check=True)
            
            # Rename output
            # train.py produces 'models/agent1.onnx' (from export_to_onnx default)
            # Actually train.py calls export_to_onnx(model_name="agent1")
            # Wait, export_to_onnx func in train.py might use "agent1" string.
            # Let's check train.py: `export_to_onnx(trainer.model, feat_dim)` -> default name="agent1"
            
            src = config.MODELS_DIR / "agent1.onnx"
            dst = ensemble_dir / f"agent1_v{i}.onnx"
            
            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"✅ Saved Member {i} to {dst}")
            else:
                print(f"❌ Error: Model output {src} not found!")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for member {i}: {e}")
            sys.exit(1)
            
    print("\n🎉 Ensemble Training Complete.")
    print(f"Models saved in {ensemble_dir}")

if __name__ == "__main__":
    train_ensemble()
