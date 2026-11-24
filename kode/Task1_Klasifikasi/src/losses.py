"""
Loss functions for Task 1: Multimodal Defect Classification
Handles:
1. Class imbalance: Weighted Cross-Entropy Loss (inversely proportional to class frequencies)
2. Task (d) Robust Label Noise handling:
   - Label Smoothing Cross-Entropy Loss
   - Bootstrapping Loss (Reed et al., 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class BootstrappingLoss(nn.Module):
    """
    Soft Bootstrapping Loss for robust learning with noisy labels (Reed et al., 2014).
    L_soft = - sum_k [ (beta * y_k + (1 - beta) * p_k) * log(p_k) ]
    Allows the model to learn from its own reliable predictions rather than overfitting noisy supervisor labels.
    """
    def __init__(self, beta=0.8, class_weights=None):
        super().__init__()
        self.beta = beta
        self.class_weights = class_weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        
        # One-hot representation of noisy supervisor target
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Convex combination of supervisor target and model belief
        soft_targets = self.beta * targets_one_hot + (1.0 - self.beta) * probs.detach()
        
        # Cross entropy with soft targets
        loss_per_sample = -torch.sum(soft_targets * log_probs, dim=-1)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(targets.device)
            sample_weights = weights[targets]
            loss_per_sample = loss_per_sample * sample_weights
            
        return torch.mean(loss_per_sample)


def get_loss_function(loss_type='weighted_ce', class_weights=None, label_smoothing=0.0, beta=0.8):
    """Factory function for loss functions."""
    loss_type = loss_type.lower()
    if loss_type in ['ce', 'cross_entropy']:
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type in ['weighted_ce', 'weighted']:
        return WeightedCrossEntropyLoss(class_weights=class_weights, label_smoothing=0.0)
    elif loss_type in ['label_smoothing', 'ls']:
        return WeightedCrossEntropyLoss(class_weights=class_weights, label_smoothing=label_smoothing if label_smoothing > 0 else 0.1)
    elif loss_type in ['bootstrap', 'bootstrapping', 'soft_bootstrap']:
        return BootstrappingLoss(beta=beta, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
