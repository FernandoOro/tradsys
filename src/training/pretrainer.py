import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from pathlib import Path

from src.config import config
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

class PreTrainer:
    """
    Handles Self-Supervised Pre-Training.
    Objective: Reconstruct masked time steps.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-4, device: str = None):
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none') # No reduction to apply mask
        
    def train(self, data_loader: DataLoader, epochs: int = 10, save_name: str = "pretrained_encoder.pth"):
        logger.info(f"Starting Pre-Training on {self.device}...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for batch_idx, (masked_x, time_x, original_x, mask) in enumerate(data_loader):
                masked_x = masked_x.to(self.device)
                time_x = time_x.to(self.device)
                original_x = original_x.to(self.device)
                mask = mask.to(self.device) # (B, L)
                
                self.optimizer.zero_grad()
                
                # Forward Pass
                # Model outputs full sequence reconstruction usually if it's Seq2Seq or Encoder
                # TransformerAgent output: (B, S, D) ??
                # Wait, TransformerAgent currently outputs (B, 2) head.
                # We need access to the SEQUENCE output for reconstruction.
                # Assumption: Model has a 'forward_encoder' or we modify it to return seq if requested.
                # Let's assume for this Phase we call 'model.encode(x)' or similar, 
                # or we add a Reconstruction Head.
                
                # Ideally:
                # enc_out = model.encoder(masked_x, time_x)
                # recon = model.recon_head(enc_out)
                
                # For compatibility with existing 'TransformerAgent', let's assume we modify it 
                # or we cast it here. 
                # If Agent is strictly Class/Reg head, we can't pretrain it without a recon head.
                # Hack: Use the transformer_encoder directly if accessible.
                
                if hasattr(self.model, 'transformer_encoder'):
                     # Embed
                     # This duplicates logic from Agent, risky.
                     # Best Design: Agent has 'forward_features' method.
                     
                     # Let's try to just run forward and assume we added a mode?
                     # No, let's implement a wrapper or ad-hoc forward here for the Transformer.
                     
                     # Access internal modules
                     src = self.model.feature_embed(masked_x)
                     pos = self.model.time_embed(time_x)
                     src = src + pos
                     output = self.model.transformer_encoder(src) # (B, S, d_model)
                     
                     # Simple Linear Projection back to Input Dim
                     # We need a projection layer. Let's create one ad-hoc if not in model.
                     if not hasattr(self, 'projection'):
                         self.projection = nn.Linear(output.shape[-1], original_x.shape[-1]).to(self.device)
                         self.optimizer.add_param_group({'params': self.projection.parameters()})
                         
                     recon = self.projection(output) # (B, S, Feat_Dim)
                else:
                    logger.error("Model incompatible with PreTraining (No accessible encoder).")
                    return

                # Calculate Loss ONLY on Masked tokens
                loss = self.criterion(recon, original_x) # (B, S, F)
                
                # Mask is (B, S). Expand to (B, S, F)
                mask_expanded = mask.unsqueeze(-1).expand_as(loss)
                
                masked_loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
                
                
                masked_loss.backward()
                self.optimizer.step()
                
                total_loss += masked_loss.item()
                count += 1
                
                if batch_idx % 100 == 0:
                    print(f"DEBUG: PreTrain Batch {batch_idx} | Loss: {masked_loss.item():.4f}")
                
            avg_loss = total_loss / count
            logger.info(f"Epoch {epoch+1}/{epochs} | PreTrain Loss: {avg_loss:.6f}")
            
        # Save Weights
        save_path = config.MODELS_DIR / save_name
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Pre-Trained Weights saved to {save_path}")

