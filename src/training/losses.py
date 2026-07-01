"""
Consciousness-aware loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConsciousnessLoss(nn.Module):
    """
    Combined loss function that includes consciousness metrics.
    """
    
    def __init__(self,
                 language_weight: float = 1.0,
                 consciousness_weight: float = 0.1,
                 workspace_efficiency_weight: float = 0.05,
                 hot_confidence_weight: float = 0.05):
        """
        Args:
            language_weight: Weight for language modeling loss
            consciousness_weight: Weight for consciousness metric loss
            workspace_efficiency_weight: Weight for workspace efficiency
            hot_confidence_weight: Weight for HOT confidence loss
        """
        super().__init__()
        
        self.language_weight = language_weight
        self.consciousness_weight = consciousness_weight
        self.workspace_efficiency_weight = workspace_efficiency_weight
        self.hot_confidence_weight = hot_confidence_weight
        
        # Language modeling loss
        self.lm_loss = nn.CrossEntropyLoss()
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                consciousness_metrics: Optional[Dict] = None,
                workspace_utilization: Optional[float] = None,
                hot_confidence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            targets: Ground truth tokens (batch_size, seq_len)
            consciousness_metrics: Optional consciousness metrics dict
            workspace_utilization: Optional workspace utilization score
            hot_confidence: Optional HOT confidence scores
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Language modeling loss (main objective)
        lm_loss = self.lm_loss(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        losses = {"lm_loss": lm_loss}
        total_loss = self.language_weight * lm_loss
        
        # Consciousness metric losses
        if consciousness_metrics is not None:
            # Encourage integration (unified consciousness)
            if "integration_current" in consciousness_metrics:
                integration = consciousness_metrics["integration_current"]
                integration_loss = (1.0 - integration)  # Minimize divergence from unified state
                losses["integration_loss"] = torch.tensor(integration_loss)
                total_loss = total_loss + self.consciousness_weight * integration_loss
            
            # Encourage differentiation (specialized processing)
            if "differentiation_current" in consciousness_metrics:
                differentiation = consciousness_metrics["differentiation_current"]
                diff_loss = (1.0 - differentiation)  # Encourage specialization
                losses["differentiation_loss"] = torch.tensor(diff_loss)
                total_loss = total_loss + self.consciousness_weight * 0.3 * diff_loss
        
        # Workspace efficiency loss
        if workspace_utilization is not None:
            # Encourage efficient workspace use (not too high, not too low)
            optimal_utilization = 0.6  # 60% utilization is ideal
            util_loss = (workspace_utilization - optimal_utilization) ** 2
            losses["workspace_efficiency_loss"] = torch.tensor(util_loss)
            total_loss = total_loss + self.workspace_efficiency_weight * util_loss
        
        # HOT confidence loss
        if hot_confidence is not None:
            # Encourage high confidence on salient tokens, low on padding
            conf_loss = F.mse_loss(hot_confidence, torch.ones_like(hot_confidence) * 0.7)
            losses["hot_confidence_loss"] = conf_loss
            total_loss = total_loss + self.hot_confidence_weight * conf_loss
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def get_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get the weighted total loss."""
        return loss_dict.get("total_loss", loss_dict.get("lm_loss", 0.0))
