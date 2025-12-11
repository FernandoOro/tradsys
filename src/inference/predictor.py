import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)

class Predictor:
    """
    Inference Engine using ONNX Runtime.
    Optimized for CPU usage on VPS.
    """
    def __init__(self, model_name: str = "agent1"):
        model_path = config.MODELS_DIR / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX Model not found at {model_path}")
            
        logger.info(f"Loading ONNX model from {model_path}...")
        # Create Inference Session
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # Get Input Names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Model loaded. Inputs: {self.input_names}, Outputs: {self.output_names}")

    def predict(self, features: np.ndarray, time_features: np.ndarray):
        """
        Runs inference.
        Args:
            features: Numpy array of shape (Batch, Seq, Feat)
            time_features: Numpy array of shape (Batch, Seq, 4)
        """
        # Ensure types (ONNX typically expects Float32)
        features = features.astype(np.float32)
        time_features = time_features.astype(np.float32)
        
        # Bind inputs
        inputs = {
            self.input_names[0]: features,
            self.input_names[1]: time_features
        }
        
        # Run
        outputs = self.session.run(self.output_names, inputs)
        
        # Output 0 is usually [Direction, Confidence]
        # Shape: (Batch, 2)
        pred_vector = outputs[0]
        
        return {
            "direction": pred_vector[:, 0], # Tanh (-1 to 1)
            "confidence": pred_vector[:, 1] # Sigmoid/Regression (0 to 1)
        }
