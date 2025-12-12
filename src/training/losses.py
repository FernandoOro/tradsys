import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon).
    
    Formula:
        L = - sum_{i \in I} \frac{1}{|P(i)|} sum_{p \in P(i)} log \frac{exp(z_i \cdot z_p / \tau)}{sum_{a \in A(i)} exp(z_i \cdot z_a / \tau)}
        
    Concept:
        - "Pulls" together points of the same class (Positives).
        - "Pushes" apart points of different classes (Negatives).
        - Projects onto a hypersphere (L2 Norm).
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of size [batch_size, n_views, ...].
            labels: ground truth of size [batch_size].
            mask: contrastive mask of size [batch_size, batch_size],
                  mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        # Check bounds
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...], '
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits (Cosine Similarity)
        # z_i . z_j / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
            
        # For numerical stability (Max Trick)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        # We don't want the model to learn that "Sample A is similar to Sample A". That's trivial.
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute Log-Prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute Mean of log-likelihood over positive
        # sum(P(i))_j * log_prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
