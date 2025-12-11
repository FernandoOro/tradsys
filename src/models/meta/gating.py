import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    """
    The Orchestrator (Meta-Learner).
    Decides which agent to trust based on market context.
    """
    def __init__(self, context_dim: int, num_agents: int = 2, hidden_dim: int = 32):
        super().__init__()
        
        # Context Encoder (Processes HMM state + Market Features)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents) # Output Logits for each agent
        )
        
    def forward(self, expert_preds: torch.Tensor, context: torch.Tensor) -> dict:
        """
        Args:
            expert_preds: Tensor (Batch, Num_Agents) containing scalar predictions from each agent.
            context: Tensor (Batch, Context_Dim) e.g., [HMM_State_Prob, Volatility, Trend_Str].
            
        Returns:
            dict: {
                'weights': Agent weights (Batch, Num_Agents),
                'final_pred': Weighted prediction (Batch, 1)
            }
        """
        # Calculate raw gating logits from context
        gating_logits = self.context_net(context)
        
        # Softmax to get probabilities (sum to 1)
        weights = F.softmax(gating_logits, dim=1) # (Batch, Num_Agents)
        
        # Weighted Sum
        # expert_preds: (Batch, Num_Agents)
        # weights: (Batch, Num_Agents)
        # Element-wise multiply + Sum
        final_pred = torch.sum(weights * expert_preds, dim=1, keepdim=True)
        
        return {
            'weights': weights,
            'final_pred': final_pred
        }
