import torch
import torch.nn as nn
import math

class TimeTimeSeriesEmbedding(nn.Module):
    """
    Cyclical Time Encoding (Sine/Cosine) for temporal features.
    Encodes constraints like Hour of Day, Day of Week, etc.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Simple Linear projection to d_model for the time features
        # Assuming inputs are [hour_sin, hour_cos, day_sin, day_cos] -> size 4
        self.time_proj = nn.Linear(4, d_model)

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
           x_time: Tensor of shape (Batch, Seq_Len, 4) containing [hour_sin, hour_cos, day_sin, day_cos]
        Returns:
           Tensor of shape (Batch, Seq_Len, d_model)
        """
        return self.time_proj(x_time)


class TransformerAgent(nn.Module):
    """
    Agent 1: Trend Follower using Native PyTorch TransformerEncoder.
    Strictly NO HuggingFace transformers.
    """
    def __init__(self, 
                 input_dim: int, 
                 d_model: int = 64, 
                 nhead: int = 4, 
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Input Embedding (Linear projection of features to d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Time Embedding
        self.time_embedding = TimeTimeSeriesEmbedding(d_model)
        
        # 3. Positional Encoding (Standard Sinusoidal)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. Transformer Encoder (NATIVE PYTORCH)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 5. Output Heads (Multi-Task)
        # Main: Classification (Trend Direction: Up/Down/Neutral? Or Binary?)
        # Context says: "Signal_Direction (-1 a 1), Confidence (0 a 1)"
        # Use Tanh for Direction (-1 to 1) and Sigmoid for Confidence (0 to 1)
        self.head_direction = nn.Linear(d_model, 1) # Output scalar -1 to 1
        self.head_confidence = nn.Linear(d_model, 1) # Output scalar 0 to 1

    @property
    def device(self):
        return next(self.parameters()).device

    def feature_embed(self, x):
        return self.input_proj(x)

    def time_embed(self, x):
        return self.time_embedding(x)

    def forward(self, x_features: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_features: (Batch, Seq_Len, input_dim) - Market features
            x_time: (Batch, Seq_Len, 4) - Time cyclical features
        """
        # Embeddings
        emb_features = self.input_proj(x_features)
        emb_time = self.time_embedding(x_time)
        
        # Combine (Simple addition or concat? Addition is standard in Transformers if mapped to same d_model)
        src = emb_features + emb_time
        
        # Positional Encoding
        src = self.pos_encoder(src)
        
        # Transformer Pass
        # output shape: (Batch, Seq_Len, d_model)
        output = self.transformer_encoder(src)
        
        # Global Pooling (Take last token representation for sequence classification/regression)
        # Alternatively use mean pooling. Let's use Last Token for causal prediction.
        last_token = output[:, -1, :] 
        
        # Heads
        direction = torch.tanh(self.head_direction(last_token))
        confidence = torch.sigmoid(self.head_confidence(last_token))
        
        # Return concatenated vector [direction, confidence]
        return torch.cat([direction, confidence], dim=1)


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding injected into the model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_first]
        """
        # Adjust implementation for batch_first=True
        # x shape: [Batch, Seq_Len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
