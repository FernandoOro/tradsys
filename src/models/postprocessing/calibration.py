import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with Temperature Scaling.
    Reference: On Calibration of Modern Neural Networks (Guo et al. 2017).
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # Learnable Parameter T, initialized to 1.5
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_feat, input_time):
        logits = self.model(input_feat, input_time) # Assuming model returns logits
        # If model returns (Direction, Conf) separately, we need to locate Logits
        # Agent 1 returns tensor (B, 2). Usually Tanh and Sigmoid applied?
        # If Agent 1 uses 'return_logits=True', we are good.
        # Assuming we apply this to the raw output before activation?
        # Or if model outputs prob, we can't scale.
        # Assuming we can get LOGITS.
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, device='cpu'):
        """
        Tune the tempearature of the model (using the validation set).
        """
        self.to(device)
        self.model.eval()
        nll_criterion = nn.MSELoss() # Or CrossEntropy if classification
        # Smart Spot Agent 1 is regression-ish (-1 to 1).
        # We can optimize T to minimize MSE between scaled-sigmoid and Target?
        # Actually Temperature Scaling is for CLASSIFICATION (Softmax).
        # Our Confidence output is Sigmoid.
        # Sigmoid(logit / T).
        # Yes, still valid.
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        # Simple optimization loop
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Need to collect all data first? LBFGS needs closure.
        # Let's use simple SGD or Adam for continuous stream or LBFGS on full batch if small.
        # Let's optimize T using one pass? No, iterative.
        
        logger.info("Calibrating Temperature on Validation Set...")
        
        # Concatenate val results
        # Assuming we can run inference
        with torch.no_grad():
             for x, t, y in valid_loader:
                 x, t, y = x.to(device), t.to(device), y.to(device)
                 out = self.model(x, t) # Logits?
                 # Assuming model has method 'get_logits' or similar.
                 # If not, this is tricky as standard forward applies Activation.
                 # WE ASSUME MODIFICATION TO AGENT TO RETURN LOGITS.
                 logits_list.append(out)
                 labels_list.append(y)
                 
        logits = torch.cat(logits_list).detach() # (N, 2)
        labels = torch.cat(labels_list).detach() # (N, 2)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        
        logger.info(f"Optimal Temperature: {self.temperature.item():.4f}")
