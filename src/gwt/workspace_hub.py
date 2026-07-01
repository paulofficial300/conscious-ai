"""
Workspace Hub - Integration point between GWT workspace and specialist modules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .workspace import GlobalWorkspace


class SpecialistModule(nn.Module):
    """Individual specialist module that feeds into the GWT workspace."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 priority: float = 0.5,
                 name: str = "specialist"):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            priority: Priority score for competition
            name: Name of specialist
        """
        super().__init__()
        
        self.name = name
        self.priority = priority
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through specialist."""
        output = self.net(x)
        output = self.layer_norm(output)
        return output


class WorkspaceHub(nn.Module):
    """
    Hub coordinating specialist modules and the global workspace.
    Implements the full GWT integration.
    """
    
    def __init__(self,
                 transformer_dim: int = 768,
                 workspace_dim: int = 512,
                 workspace_capacity: int = 4,
                 num_specialists: int = 4,
                 specialist_configs: Optional[List[Dict]] = None):
        """
        Args:
            transformer_dim: Dimension from transformer output
            workspace_dim: Dimension of workspace
            workspace_capacity: Max specialists that can broadcast
            num_specialists: Total number of specialists
            specialist_configs: Optional configuration for each specialist
        """
        super().__init__()
        
        self.transformer_dim = transformer_dim
        self.workspace_dim = workspace_dim
        self.num_specialists = num_specialists
        
        # Create specialist modules
        self.specialists = nn.ModuleList()
        self.specialist_names = []
        
        if specialist_configs:
            for config in specialist_configs:
                specialist = SpecialistModule(
                    input_dim=transformer_dim,
                    output_dim=workspace_dim,
                    hidden_dim=config.get("hidden_dim", 256),
                    priority=config.get("priority", 0.5),
                    name=config.get("name", f"specialist_{len(self.specialists)}")
                )
                self.specialists.append(specialist)
                self.specialist_names.append(specialist.name)
        else:
            # Default specialists
            for i in range(num_specialists):
                specialist = SpecialistModule(
                    input_dim=transformer_dim,
                    output_dim=workspace_dim,
                    name=f"specialist_{i}"
                )
                self.specialists.append(specialist)
                self.specialist_names.append(f"specialist_{i}")
        
        # Global workspace
        self.workspace = GlobalWorkspace(
            workspace_dim=workspace_dim,
            workspace_capacity=workspace_capacity,
            num_specialists=len(self.specialists),
            specialist_input_dim=workspace_dim,
            broadcasting_mechanism="attention_based"
        )
        
        # Feedback projection from workspace back to transformer
        self.workspace_feedback = nn.Linear(workspace_dim, transformer_dim)
        
        # Statistics tracking
        self.register_buffer("specialist_activity", 
                           torch.zeros(len(self.specialists)))
    
    def forward(self,
                transformer_output: torch.Tensor,
                hot_signal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process transformer output through specialist modules and workspace.
        
        Args:
            transformer_output: Output from transformer (batch_size, seq_len, transformer_dim)
            hot_signal: Optional signal from Higher-Order Thought module
            
        Returns:
            Dictionary containing:
                - workspace_broadcast: Information broadcast from workspace
                - specialist_outputs: Outputs from each specialist
                - workspace_state: Current workspace state
                - feedback: Feedback signal to transformer
        """
        batch_size, seq_len, _ = transformer_output.shape
        
        # Process through specialist modules
        specialist_outputs = []
        for i, specialist in enumerate(self.specialists):
            output = specialist(transformer_output)
            specialist_outputs.append(output)
            self.specialist_activity[i] = output.mean().detach()
        
        # Compute competition between specialists
        competition_scores = self.workspace.compute_competition(specialist_outputs)
        
        # Broadcast winning specialists to workspace
        workspace_broadcast = self.workspace.broadcast_to_workspace(
            specialist_outputs,
            competition_scores,
            hot_signal
        )
        
        # Generate feedback signal
        feedback = self.workspace_feedback(workspace_broadcast)
        
        # Get workspace state
        workspace_state = self.workspace.get_workspace_state()
        
        return {
            "workspace_broadcast": workspace_broadcast,
            "specialist_outputs": specialist_outputs,
            "workspace_state": workspace_state,
            "feedback": feedback,
            "competition_scores": competition_scores,
        }
    
    def get_specialist_activity(self) -> Dict[str, float]:
        """Get activity level of each specialist."""
        return {
            name: activity.item()
            for name, activity in zip(self.specialist_names, self.specialist_activity)
        }
    
    def reset(self):
        """Reset workspace and activity tracking."""
        self.workspace.reset_workspace()
        self.specialist_activity.fill_(0)
