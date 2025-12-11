
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
        
        # Output Head (Binary Classification for Reversal)
        # 1 = Reversal Likely, 0 = Continuation
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
            
        out = self.input_proj(x)
        out = self.bn_in(out)
        out = self.relu(out)
        
        for block in self.blocks:
            out = block(out)
            
        return self.head(out)

    def get_encoder(self):
        """Returns the encoder part for potential transfer learning"""
        return nn.Sequential(
            self.input_proj,
            self.bn_in,
            self.relu,
            *self.blocks
        )

class ResNetAutoEncoder(nn.Module):
    """
    Pre-training Module for Agent 2.
    Type: Denoising Autoencoder (DAE).
    Goal: Learn manifold structure of 'Regime 0' data from scarce samples.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        
        # Encoder (Matches Agent Structure)
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_enc = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Decoder (Mirror Image)
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
            
        # Add Noise (Denoising)
        noise = torch.randn_like(x) * 0.1
        h = x + noise
        
        # Encode
        h = self.encoder_proj(h)
        h = self.bn_enc(h)
        h = self.relu(h)
        for block in self.encoder_blocks:
            h = block(h)
            
        # Decode
        for block in self.decoder_blocks:
            h = block(h)
        out = self.decoder_proj(h)
        
        return out
