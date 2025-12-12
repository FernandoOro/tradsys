import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from src.training.losses import SupervisedContrastiveLoss

logger = logging.getLogger(__name__)

class ProjectionHead(nn.Module):
    """
    MLP used for projecting embeddings to a hypersphere.
    Structure: Input -> Hidden -> ReLU -> L2_Norm
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        # Normalize to unit sphere
        return torch.nn.functional.normalize(x, dim=1)

class ContrastiveTrainer:
    """
    Specialized Trainer for Phase 28: Geometry of Alpha.
    Process:
    1. Encoder + Projector -> SupCon Loss
    2. Encoder (Frozen) + Classifier -> CE Loss
    """
    def __init__(self, model, lr=1e-3, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.lr = lr
        
    def train_contrastive(self, train_loader, epochs=5):
        """
        Step 1: Train the Encoder to Cluster classes geometrically.
        """
        logger.info(">>> Starting Contrastive Pre-Training (SupCon)...")
        
        # 1. Attach Projection Head
        # Assuming model.d_model is the embedding size
        embed_dim = self.model.d_model 
        projector = ProjectionHead(embed_dim, hidden_dim=embed_dim, output_dim=64).to(self.device)
        
        # Optimizer targets both Encoder and Projector
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': projector.parameters()}
        ], lr=self.lr)
        
        criterion = SupervisedContrastiveLoss(temperature=0.07)
        
        # Train Loop
        self.model.train()
        projector.train()
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            pbar = tqdm(train_loader, desc=f"SupCon Epoch {epoch+1}/{epochs}")
            for x, t, y in pbar:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                
                # y shape is (Batch, 2) [Dir, Conf]. We need Class Label (0 or 1)
                # target > 0 -> Class 1, target < 0 -> Class 0
                # y[:, 0] is direction (-1 or 1). Map -1 -> 0, 1 -> 1
                labels = (y[:, 0] > 0).float()
                
                optimizer.zero_grad()
                
                # Get Embeddings from Transformer (Not Logits)
                # We need to hack TransformerAgent to give us raw embeddings.
                # Or we assume TransformerAgent.get_embeddings() exists, or we slice the forward.
                # Ideally, we call forward, but `TransformerAgent` returns logits.
                # Let's inspect `TransformerAgent` code later. For now assume `extract_features`.
                
                # If extract_features doesn't exist, we will add it to the Protocol or Monkey Patch.
                if hasattr(self.model, 'extract_features'):
                     features = self.model.extract_features(x, t) # (Batch, d_model)
                else:
                     # Fallback: Assume the creation of this method or failure
                     raise NotImplementedError("Model must implement `extract_features` for SupCon.")

                # Project
                # Reshape for SupCon: [Batch, Views, Dim]. We have 1 View.
                proj = projector(features)
                proj = proj.unsqueeze(1) # (Batch, 1, 64)
                
                loss = criterion(proj, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                pbar.set_postfix({'loss': total_loss/count})
                
        logger.info(">>> Contrastive Pre-Training Complete.")
        
        # Remove Projector (Clean up)
        del projector
        return self.model

    def train_linear_probe(self, train_loader, val_loader, epochs=5):
        """
        Step 2: Freeze Encoder, Train Linear Classifier.
        """
        logger.info(">>> Starting Linear Probing...")
        
        # Freeze Encoder
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Re-enable Classifier Head Gradients
        # This assumes the model has a 'head' or 'fc' layer that is the classifier
        if hasattr(self.model, 'decoder'):
             for param in self.model.decoder.parameters():
                 param.requires_grad = True
        else:
             raise NotImplementedError("Model structure unknown. Cannot find classifier head.")
             
        # Optimizer only for the head
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() # Standard Binary
        
        self.model.train() # Set to train mode (Dropout might be active? Ideally eval for encoder, train for head)
        # Usually checking Linear Probing on Frozen features means Encoder in Eval mode.
        self.model.eval() 
        # But we need head in train mode?
        # Simpler: Set all to train, but grads frozen for encoder.
        # But Dropout will be active. 
        # Standard SupCon practice: Frozen Encoder (Eval), Trainable Head.
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            pbar = tqdm(train_loader, desc=f"Linear Probe Epoch {epoch+1}/{epochs}")
            for x, t, y in pbar:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                
                label_dir = (y[:, 0] > 0).float().unsqueeze(1) # (Batch, 1) usually
                
                optimizer.zero_grad()
                
                # Forward full pass (Encoder frozen, Head fluid)
                # But wait, we need to ensure the head is actually training.
                out = self.model(x, t) # (Batch, 2)
                
                # Only train Direction head (out[:, 0])
                pred_dir = out[:, 0].unsqueeze(1)
                
                loss = criterion(pred_dir, label_dir)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                pbar.set_postfix({'loss': total_loss/count})
                
            # Validation
            val_loss = self.validate(val_loader)
            logger.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            
    def validate(self, val_loader):
        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                label_dir = (y[:, 0] > 0).float().unsqueeze(1)
                out = self.model(x, t)
                loss = criterion(out[:, 0].unsqueeze(1), label_dir)
                total_loss += loss.item()
                count += 1
        return total_loss / count
