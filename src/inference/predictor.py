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
        # Support nested paths (e.g., agent2/agent2)
        if "/" not in model_name:
            # Fallback for legacy "agent1"
            model_path = config.MODELS_DIR / f"{model_name}.onnx"
        else:
             # Full relative path support
             model_path = config.MODELS_DIR / f"{model_name}.onnx"

        # Check secondary location for agent2 if not found
        if not model_path.exists() and model_name == "agent2":
             model_path = config.MODELS_DIR / "agent2" / "agent2.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX Model not found at {model_path}")
            
        logger.info(f"Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Model loaded. Inputs: {self.input_names}, Outputs: {self.output_names}")

    def predict(self, features: np.ndarray, time_features: np.ndarray = None):
        """
        Runs inference. Automatically adjusts to model inputs.
        """
        features = features.astype(np.float32)
        inputs = {}
        
        # Binding Logic
        if len(self.input_names) == 2:
            # Agent 1: Transformer (Features + Time)
            if time_features is None:
                raise ValueError("Model requires time_features but None provided.")
            inputs[self.input_names[0]] = features
            inputs[self.input_names[1]] = time_features.astype(np.float32)
            
        elif len(self.input_names) == 1:
            # Agent 2: ResNet (Features only, Flattened)
            # If input is (B, Seq, Feat), Flatten it
            if features.ndim == 3:
                B, S, F = features.shape
                features = features.reshape(B, S*F)
            inputs[self.input_names[0]] = features
            
        # Run
        outputs = self.session.run(self.output_names, inputs)
        
        # Formatting Output
        # Agent 1 (Transformer): [Direction, Confidence]
        # Agent 2 (ResNet): [ReversalScore] (Logits or Prob?)
        
        pred_vector = outputs[0]
        
        if pred_vector.shape[1] == 2:
            return {
                "direction": pred_vector[:, 0], 
                "confidence": pred_vector[:, 1]
            }
        else:
            # Agent 2 (Scalar Output)
            return {
                "score": pred_vector[:, 0]
            }
