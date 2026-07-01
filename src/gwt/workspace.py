"""
Global Workspace Theory (GWT) Implementation.
Central hub where specialist modules report findings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class GlobalWorkspace(nn.Module):
    """
    Central workspace implementing Global Workspace Theory.
    Acts as a limited-capacity information hub that broadcasts to all modules.
    """
    
    def __init__(self,
                 workspace_dim: int = 512,
                 workspace_capacity: int = 4,
                 num_specialists: int = 4,
                 specialist_input_dim: int = 768,
                 broadcasting_mechanism: str = "attention_based",
                 dropout: float = 0.1):
        """
        Args:
            workspace_dim: Dimension of the workspace
            workspace_capacity: Maximum number of specialists that can broadcast
            num_specialists: Total number of specialist modules
            specialist_input_dim: Input dimension from specialist modules
            broadcasting_mechanism: How to select winners ("attention_based", "winner_takes_all")
            dropout: Dropout rate
        """
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.workspace_capacity = workspace_capacity
        self.num_specialists = num_specialists
        self.broadcasting_mechanism = broadcasting_mechanism
        
        # Specialist projection layers
        self.specialist_projections = nn.ModuleList([
            nn.Linear(specialist_input_dim, workspace_dim)
            for _ in range(num_specialists)
        ])
        
        # Competition mechanism
        if broadcasting_mechanism == "attention_based":
            self.competition_weights = nn.Parameter(
                torch.randn(num_specialists, workspace_dim)
            )
        
        # Workspace memory and state
        self.register_buffer("workspace_state", torch.zeros(workspace_dim))
        self.register_buffer("broadcast_history", 
                           torch.zeros(num_specialists))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(workspace_dim)
    
    def compute_competition(self, 
                           specialist_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute competition scores for specialist modules.
        
        Args:
            specialist_outputs: List of outputs from each specialist
            
        Returns:
            Competition scores for each specialist
        """
        batch_size, seq_len = specialist_outputs[0].shape[:2]
        num_specialists = len(specialist_outputs)
        
        # Stack specialist outputs
        stacked = torch.stack(specialist_outputs, dim=1)
        # Shape: (batch_size, num_specialists, seq_len, workspace_dim)
        
        if self.broadcasting_mechanism == "attention_based":
            # Compute attention-based competition scores
            scores = torch.einsum('bsld,sd->bs', stacked, self.competition_weights)
            scores = scores / (self.workspace_dim ** 0.5)
            competition_scores = torch.softmax(scores, dim=1)
            
        else:  # winner_takes_all
            # Use max activation as competition score
            max_vals = torch.max(torch.abs(stacked), dim=-1)[0]
            max_vals = torch.max(max_vals, dim=-1)[0]  # Reduce sequence dimension
            competition_scores = torch.softmax(max_vals.float(), dim=1)
        
        return competition_scores
    
    def broadcast_to_workspace(self,
                              specialist_outputs: List[torch.Tensor],
                              competition_scores: torch.Tensor,
                              hot_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Broadcast winning specialists' information to workspace.
        
        Args:
            specialist_outputs: Outputs from each specialist module
            competition_scores: Competition weights for each specialist
            hot_signal: Optional Higher-Order Thought signal modulating broadcast
            
        Returns:
            Workspace broadcast content (batch_size, seq_len, workspace_dim)
        """
        batch_size, seq_len = specialist_outputs[0].shape[:2]
        
        # Project specialist outputs to workspace dimension
        projected_outputs = []
        for i, output in enumerate(specialist_outputs):
            proj = self.specialist_projections[i](output)
            projected_outputs.append(proj)
        
        # Stack for easier processing
        stacked = torch.stack(projected_outputs, dim=1)
        # Shape: (batch_size, num_specialists, seq_len, workspace_dim)
        
        # Apply competition scores
        # Reshape scores for broadcasting
        scores = competition_scores.view(batch_size, self.num_specialists, 1, 1)
        weighted = stacked * scores
        
        # Apply HOT modulation if provided
        if hot_signal is not None:
            # hot_signal: (batch_size, seq_len, workspace_dim)
            hot_mod = hot_signal.unsqueeze(1)  # Add specialist dimension
            weighted = weighted * hot_mod
        
        # Sum across specialists to create workspace broadcast
        broadcast = torch.sum(weighted, dim=1)
        # Shape: (batch_size, seq_len, workspace_dim)
        
        # Update workspace state
        self.workspace_state = broadcast.mean(dim=0).mean(dim=0).detach()
        
        # Track which specialists broadcast
        top_k = min(self.workspace_capacity, self.num_specialists)
        top_specialists = torch.topk(competition_scores, top_k, dim=1)[1]
        for i in range(self.num_specialists):
            self.broadcast_history[i] = (top_specialists == i).float().sum().item()
        
        # Normalize workspace broadcast
        broadcast = self.layer_norm(broadcast)
        broadcast = self.dropout(broadcast)
        
        return broadcast
    
    def get_workspace_state(self) -> Dict[str, torch.Tensor]:
        """Get current workspace state."""
        return {
            "workspace_state": self.workspace_state.clone(),
            "broadcast_history": self.broadcast_history.clone(),
        }
    
    def reset_workspace(self):
        """Reset workspace state for new sequence."""
        self.workspace_state.fill_(0)
        self.broadcast_history.fill_(0)
