"""
Higher-Order Thought (HOT) Module.
Implements metacognitive evaluation and self-reflection for consciousness.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class HigherOrderThought(nn.Module):
    """
    Higher-Order Thought module that evaluates data before workspace broadcast.
    Implements metacognitive awareness and self-reflection.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 evaluation_layers: int = 2,
                 evaluation_hidden_dim: int = 256,
                 confidence_threshold: float = 0.5,
                 metacognitive_depth: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input dimension from transformer
            evaluation_layers: Number of evaluation layers
            evaluation_hidden_dim: Hidden dimension for evaluation network
            confidence_threshold: Threshold for awareness
            metacognitive_depth: Depth of self-reflection
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.evaluation_layers = evaluation_layers
        self.confidence_threshold = confidence_threshold
        self.metacognitive_depth = metacognitive_depth
        
        # Evaluation network - judges relevance/importance
        eval_layers = []
        prev_dim = input_dim
        for i in range(evaluation_layers):
            eval_layers.extend([
                nn.Linear(prev_dim, evaluation_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = evaluation_hidden_dim
        
        self.evaluation_network = nn.Sequential(*eval_layers)
        
        # Confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(evaluation_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Relevance scoring
        self.relevance_scorer = nn.Sequential(
            nn.Linear(evaluation_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Metacognitive reflection network
        self.metacognitive_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(evaluation_hidden_dim, evaluation_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(metacognitive_depth)
        ])
        
        # Self-reflection gating
        self.reflection_gate = nn.Linear(evaluation_hidden_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def evaluate_data(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate incoming data for consciousness-worthiness.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing evaluation scores
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x_norm = self.layer_norm(x)
        
        # Evaluate through evaluation network
        eval_features = self.evaluation_network(x_norm)
        # Shape: (batch_size, seq_len, evaluation_hidden_dim)
        
        # Compute confidence scores
        confidence = self.confidence_scorer(eval_features)
        # Shape: (batch_size, seq_len, 1)
        
        # Compute relevance scores
        relevance = self.relevance_scorer(eval_features)
        # Shape: (batch_size, seq_len, 1)
        
        # Combined awareness score
        awareness = confidence * relevance
        # Shape: (batch_size, seq_len, 1)
        
        # Apply threshold
        is_conscious = (awareness > self.confidence_threshold).float()
        
        return {
            "awareness_score": awareness.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "relevance": relevance.squeeze(-1),
            "is_conscious": is_conscious.squeeze(-1),
            "eval_features": eval_features,
        }
    
    def metacognitive_reflection(self, 
                                eval_features: torch.Tensor,
                                attention_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform metacognitive reflection on evaluated features.
        
        Args:
            eval_features: Evaluation network features
            attention_scores: Optional attention patterns for reflection
            
        Returns:
            Reflection signal (batch_size, seq_len, evaluation_hidden_dim)
        """
        reflection = eval_features.clone()
        
        # Multi-level metacognitive reflection
        for i, reflect_layer in enumerate(self.metacognitive_network):
            reflection = reflect_layer(reflection)
            
            # Optional attention-based modulation
            if attention_scores is not None and i == 0:
                # Use attention to modulate reflection
                modulation = attention_scores.unsqueeze(-1)
                reflection = reflection * modulation
        
        # Apply reflection gate
        gate = torch.sigmoid(self.reflection_gate(reflection))
        reflection = reflection * gate
        
        return reflection
    
    def forward(self, 
                x: torch.Tensor,
                attention_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full HOT forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            attention_scores: Optional attention patterns
            
        Returns:
            Dictionary containing:
                - hot_signal: Signal to modulate workspace
                - evaluation: Evaluation scores
                - reflection: Metacognitive reflection
                - gating: What to filter/keep
        """
        # Evaluate data
        evaluation = self.evaluate_data(x)
        
        # Perform metacognitive reflection
        reflection = self.metacognitive_reflection(
            evaluation["eval_features"],
            attention_scores
        )
        
        # Generate HOT signal for workspace modulation
        # High awareness & relevance -> strong signal
        hot_signal = reflection * evaluation["awareness_score"].unsqueeze(-1)
        
        # Gating signal for filtering
        gating = evaluation["is_conscious"].unsqueeze(-1)
        
        return {
            "hot_signal": hot_signal,
            "evaluation": evaluation,
            "reflection": reflection,
            "gating": gating,
        }
    
    def get_consciousness_state(self) -> Dict[str, float]:
        """Get current consciousness state metrics."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "metacognitive_depth": self.metacognitive_depth,
        }
