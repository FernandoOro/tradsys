import torch
import numpy as np

class GaussianNoise:
    """
    Injects random Gaussian noise into the input tensor.
    Regularization technique to prevent overfitting.
    """
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise
