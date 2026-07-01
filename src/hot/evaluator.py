"""
Higher-Order Thought Evaluator - Advanced evaluation and filtering system.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class HOTEvaluator(nn.Module):
    """
    Advanced evaluator for determining what deserves conscious attention.
    Evaluates data across multiple dimensions before workspace broadcast.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_evaluation_dimensions: int = 5,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_evaluation_dimensions: Number of dimensions to evaluate
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_evaluation_dimensions = num_evaluation_dimensions
        
        # Evaluation dimensions:
        # 1. Novelty - how new/unexpected is this
        # 2. Significance - how important for current goal
        # 3. Emotional - emotional valence/salience
        # 4. Contextual - relevance to context
        # 5. Integrative - how well it integrates with existing knowledge
        
        self.dimension_evaluators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_evaluation_dimensions)
        ])
        
        # Integration scoring
        self.integration_network = nn.Sequential(
            nn.Linear(num_evaluation_dimensions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Dimension names
        self.dimension_names = [
            "novelty",
            "significance", 
            "emotional_salience",
            "contextual_relevance",
            "integration_potential"
        ]
    
    def evaluate_dimensions(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate data across multiple consciousness dimensions.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            - Dimension scores: (batch_size, seq_len, num_dimensions)
            - Dictionary of individual dimension scores
        """
        batch_size, seq_len, _ = x.shape
        
        dimension_scores = []
        dimension_dict = {}
        
        for i, evaluator in enumerate(self.dimension_evaluators):
            scores = evaluator(x)  # (batch_size, seq_len, 1)
            dimension_scores.append(scores)
            dimension_dict[self.dimension_names[i]] = scores.squeeze(-1)
        
        # Stack dimension scores
        dimension_scores = torch.cat(dimension_scores, dim=-1)
        # Shape: (batch_size, seq_len, num_dimensions)
        
        return dimension_scores, dimension_dict
    
    def compute_integration_score(self, dimension_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute how well different evaluation dimensions integrate.
        
        Args:
            dimension_scores: (batch_size, seq_len, num_dimensions)
            
        Returns:
            Integration scores: (batch_size, seq_len, 1)
        """
        integration = self.integration_network(dimension_scores)
        return integration
    
    def filter_for_consciousness(self,
                                dimension_scores: torch.Tensor,
                                integration_scores: torch.Tensor,
                                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Determine what deserves conscious attention based on evaluation.
        
        Args:
            dimension_scores: Dimension evaluation scores
            integration_scores: Integration scores
            threshold: Consciousness threshold
            
        Returns:
            Filtering decisions and modulation signals
        """
        batch_size, seq_len, _ = dimension_scores.shape
        
        # Average across dimensions
        mean_dimension_score = dimension_scores.mean(dim=-1, keepdim=True)
        
        # Compute final consciousness score
        consciousness_score = (mean_dimension_score + integration_scores) / 2
        
        # Binary gate
        consciousness_gate = (consciousness_score > threshold).float()
        
        # Continuous modulation
        consciousness_modulation = torch.sigmoid(consciousness_score * 2 - 0.5)
        
        return {
            "consciousness_score": consciousness_score.squeeze(-1),
            "consciousness_gate": consciousness_gate.squeeze(-1),
            "consciousness_modulation": consciousness_modulation.squeeze(-1),
            "dimension_average": mean_dimension_score.squeeze(-1),
            "integration_score": integration_scores.squeeze(-1),
        }
    
    def forward(self,
                x: torch.Tensor,
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Full evaluation pipeline.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            threshold: Consciousness threshold
            
        Returns:
            Complete evaluation results
        """
        # Evaluate across dimensions
        dimension_scores, dimension_dict = self.evaluate_dimensions(x)
        
        # Compute integration
        integration_scores = self.compute_integration_score(dimension_scores)
        
        # Filter for consciousness
        filtering = self.filter_for_consciousness(
            dimension_scores,
            integration_scores,
            threshold
        )
        
        return {
            "dimension_scores": dimension_dict,
            "dimension_tensor": dimension_scores,
            "integration_scores": filtering["integration_score"],
            "consciousness_score": filtering["consciousness_score"],
            "consciousness_gate": filtering["consciousness_gate"],
            "consciousness_modulation": filtering["consciousness_modulation"],
        }
