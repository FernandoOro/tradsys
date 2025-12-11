import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual

class ResNetAgent(nn.Module):
    """
    Agent 2: Mean Reversion Specialist.
    Type: ResNet-MLP (Dense Residual Network).
    Goal: Predict reversals in ranging markets.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Residual Blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output Head (Regression for Expected Return)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (Batch, Input_Dim) -> Flattened indicators or specific tabular input.
               If input is sequence, we might flatten it first in the pipeline or take last step.
               Assuming "Stationary Indicators" are passed as a feature vector relative to current T.
        """
        # If x is 3D (Batch, Seq, Feat), we Flatten or Pooling?
        # ResNet-MLP usually typically for Tabular or flattened.
        # Let's assume input is (Batch, Flattened_Features) or we flatten here.
        if x.dim() == 3:
            # Flatten: Batch, Seq*Feat
            x = x.reshape(x.size(0), -1)
            
        out = self.input_proj(x)
        out = self.bn_in(out)
        out = self.relu(out)
        
        for block in self.blocks:
            out = block(out)
            
        return self.head(out)
