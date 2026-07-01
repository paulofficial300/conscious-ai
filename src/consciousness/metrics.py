"""
Consciousness Metrics - Measures consciousness-like properties of the model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np


class ConsciousnessMetrics:
    """
    Computes various metrics related to consciousness properties.
    """
    
    def __init__(self, moving_avg_window: int = 100):
        """
        Args:
            moving_avg_window: Window size for moving averages
        """
        self.moving_avg_window = moving_avg_window
        self.history = {
            "integration": [],
            "differentiation": [],
            "workspace_utilization": [],
            "hot_confidence": [],
            "broadcast_entropy": [],
        }
    
    def compute_integration(self, hidden_states: list) -> float:
        """
        Compute integration of information across layers.
        Higher integration = more unified/conscious state.
        
        Args:
            hidden_states: List of hidden states from each layer
            
        Returns:
            Integration score (0-1)
        """
        if len(hidden_states) < 2:
            return 0.0
        
        # Convert to tensors
        states = [h.detach().cpu().numpy() if torch.is_tensor(h) else h 
                  for h in hidden_states]
        
        # Compute correlations between consecutive layers
        correlations = []
        for i in range(len(states) - 1):
            s1 = states[i].mean(axis=0).mean(axis=0)  # Average over batch and sequence
            s2 = states[i + 1].mean(axis=0).mean(axis=0)
            
            if len(s1) > 0 and len(s2) > 0:
                corr = np.corrcoef(s1, s2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        # Integration is average correlation (how unified the states are)
        integration = np.mean(correlations)
        return max(0.0, min(1.0, integration))  # Clamp to [0, 1]
    
    def compute_differentiation(self, attention_scores: list) -> float:
        """
        Compute differentiation - how much different attention patterns emerge.
        Higher differentiation = more specialized processing.
        
        Args:
            attention_scores: List of attention weights from layers
            
        Returns:
            Differentiation score (0-1)
        """
        if not attention_scores:
            return 0.0
        
        # Compute entropy of attention distributions
        entropies = []
        for attn in attention_scores:
            if torch.is_tensor(attn):
                attn = attn.detach().cpu().numpy()
            
            # Average attention across batch and heads
            avg_attn = attn.mean(axis=0).mean(axis=0)  # (seq_len, seq_len)
            
            # Compute entropy for each query position
            for row in avg_attn:
                if row.sum() > 0:
                    row = row / row.sum()
                    entropy = -np.sum(row * np.log(row + 1e-10))
                    entropies.append(entropy)
        
        if not entropies:
            return 0.0
        
        # Normalize entropy by max possible (log(seq_len))
        max_entropy = np.log(avg_attn.shape[0])
        avg_entropy = np.mean(entropies)
        
        differentiation = avg_entropy / (max_entropy + 1e-10)
        return max(0.0, min(1.0, differentiation))
    
    def compute_workspace_utilization(self, specialist_activity: Dict[str, float]) -> float:
        """
        Compute how much of the workspace capacity is being utilized.
        
        Args:
            specialist_activity: Activity levels of specialists
            
        Returns:
            Utilization score (0-1)
        """
        if not specialist_activity:
            return 0.0
        
        activities = list(specialist_activity.values())
        
        # Utilization is proportion of active specialists
        num_active = sum(1 for a in activities if a > 0.5)
        utilization = num_active / max(len(activities), 1)
        
        return utilization
    
    def compute_hot_confidence(self, evaluation_scores: Dict[str, torch.Tensor]) -> float:
        """
        Compute confidence of Higher-Order Thought module.
        
        Args:
            evaluation_scores: Confidence and relevance scores from HOT
            
        Returns:
            Average confidence score
        """
        if not evaluation_scores:
            return 0.5
        
        if "confidence" in evaluation_scores:
            conf = evaluation_scores["confidence"]
            if torch.is_tensor(conf):
                conf = conf.detach().cpu().numpy()
            return float(np.mean(conf))
        
        return 0.5
    
    def compute_broadcast_entropy(self, competition_scores: torch.Tensor) -> float:
        """
        Compute entropy of specialist competition scores.
        Lower entropy = more concentrated broadcasts (sharp choices).
        
        Args:
            competition_scores: Softmax scores from specialist competition
            
        Returns:
            Entropy score (0-1, normalized)
        """
        if torch.is_tensor(competition_scores):
            scores = competition_scores.detach().cpu().numpy()
        else:
            scores = competition_scores
        
        # Average across batch
        scores = scores.mean(axis=0)
        
        # Ensure probabilities sum to 1
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            return 0.0
        
        # Compute entropy
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        return float(normalized_entropy)
    
    def update(self, 
               hidden_states: Optional[list] = None,
               attention_scores: Optional[list] = None,
               specialist_activity: Optional[Dict[str, float]] = None,
               evaluation_scores: Optional[Dict[str, torch.Tensor]] = None,
               competition_scores: Optional[torch.Tensor] = None):
        """Update all metrics."""
        
        if hidden_states is not None:
            integration = self.compute_integration(hidden_states)
            self.history["integration"].append(integration)
        
        if attention_scores is not None:
            diff = self.compute_differentiation(attention_scores)
            self.history["differentiation"].append(diff)
        
        if specialist_activity is not None:
            util = self.compute_workspace_utilization(specialist_activity)
            self.history["workspace_utilization"].append(util)
        
        if evaluation_scores is not None:
            conf = self.compute_hot_confidence(evaluation_scores)
            self.history["hot_confidence"].append(conf)
        
        if competition_scores is not None:
            entropy = self.compute_broadcast_entropy(competition_scores)
            self.history["broadcast_entropy"].append(entropy)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values (using moving average)."""
        metrics = {}
        
        for key, values in self.history.items():
            if values:
                # Use last N values for moving average
                window = min(len(values), self.moving_avg_window)
                avg = np.mean(values[-window:])
                metrics[f"{key}_current"] = avg
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, list]:
        """Get full history of all metrics."""
        return self.history
    
    def reset(self):
        """Reset all metrics."""
        for key in self.history:
            self.history[key] = []
