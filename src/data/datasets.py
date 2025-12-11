import torch
from torch.utils.data import Dataset
import numpy as np

class MaskedTimeSeriesDataset(Dataset):
    """
    Dataset for Self-Supervised Learning (Masked Modeling).
    """
    def __init__(self, data: np.ndarray, time_features: np.ndarray, seq_len: int = 64, mask_prob: float = 0.15):
        """
        Args:
            data: Normalized features (N, Feats).
            time_features: Time embeddings (N, 4).
            seq_len: Sequence length.
            mask_prob: Probability of masking a time step.
        """
        self.data = torch.FloatTensor(data)
        self.time_features = torch.FloatTensor(time_features)
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Extract window
        # Shape: (Seq_Len, Feats)
        window_data = self.data[idx : idx + self.seq_len].clone()
        window_time = self.time_features[idx : idx + self.seq_len].clone()
        
        # Create Mask
        # Shape: (Seq_Len,)
        mask = torch.rand(self.seq_len) < self.mask_prob
        
        # Apply Masking Policies
        masked_input = window_data.clone()
        
        # Get indices derived from mask
        mask_indices = torch.where(mask)[0]
        
        for idx in mask_indices:
            prob = torch.rand(1).item()
            if prob < 0.8:
                # Replace with Zero (or explicit token if categorical, here numerical -> 0)
                masked_input[idx] = 0.0
            elif prob < 0.9:
                # Replace with Random Noise
                masked_input[idx] = torch.randn_like(masked_input[idx])
            else:
                # Keep same (Identity)
                pass
                
        # Targets are the original values at the masked positions
        # Actually, we usually return full original sequence and perform loss masking in the loop
        # But returning just mask helps.
        
        return masked_input, window_time, window_data, mask
