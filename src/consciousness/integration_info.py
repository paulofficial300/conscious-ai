"""
Integrated Information Theory (IIT) based consciousness measurement.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class IntegratedInformation:
    """
    Compute Integrated Information (Phi) - a measure of consciousness
    based on Integrated Information Theory.
    """
    
    def __init__(self, num_states: int = 10):
        """
        Args:
            num_states: Number of quantized states for discretization
        """
        self.num_states = num_states
    
    def discretize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Discretize continuous state into discrete values.
        
        Args:
            state: Continuous state tensor
            
        Returns:
            Discretized state
        """
        state_flat = state.flatten()
        state_min = state_flat.min()
        state_max = state_flat.max()
        
        if state_max == state_min:
            return np.zeros_like(state, dtype=int)
        
        # Quantize to [0, num_states)
        normalized = (state_flat - state_min) / (state_max - state_min + 1e-10)
        discretized = np.floor(normalized * self.num_states).astype(int)
        discretized = np.minimum(discretized, self.num_states - 1)
        
        return discretized.reshape(state.shape)
    
    def compute_entropy(self, distribution: np.ndarray) -> float:
        """
        Compute Shannon entropy.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy in bits
        """
        distribution = distribution / (distribution.sum() + 1e-10)
        return -np.sum(distribution * np.log2(distribution + 1e-10))
    
    def partition_system(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partition system into two subsystems.
        
        Args:
            state: State vector
            
        Returns:
            Two subsystem states
        """
        mid = len(state) // 2
        return state[:mid], state[mid:]
    
    def compute_mutual_information(self, 
                                   state_full: np.ndarray,
                                   state_part1: np.ndarray,
                                   state_part2: np.ndarray) -> float:
        """
        Compute mutual information between partitions.
        
        Args:
            state_full: Full system state
            state_part1: First partition
            state_part2: Second partition
            
        Returns:
            Mutual information
        """
        # Compute entropies
        h_full = self.compute_entropy(np.bincount(state_full))
        h_part1 = self.compute_entropy(np.bincount(state_part1))
        h_part2 = self.compute_entropy(np.bincount(state_part2))
        
        # Mutual information (simplified)
        mi = (h_part1 + h_part2) - h_full
        return max(0.0, mi)
    
    def compute_phi(self, hidden_state: torch.Tensor) -> float:
        """
        Compute integrated information (Phi).
        
        Args:
            hidden_state: Hidden state tensor from model
            
        Returns:
            Integrated information score (0-1 normalized)
        """
        if torch.is_tensor(hidden_state):
            state = hidden_state.detach().cpu().numpy()
        else:
            state = hidden_state
        
        # Flatten to 1D
        state_flat = state.flatten()
        
        # Discretize
        state_discrete = self.discretize_state(state_flat)
        
        # Partition
        part1, part2 = self.partition_system(state_discrete)
        
        # Compute mutual information
        mi = self.compute_mutual_information(state_discrete, part1, part2)
        
        # Normalize by max possible information
        max_info = np.log2(self.num_states)
        phi = mi / (max_info + 1e-10)
        
        return float(phi)
    
    def compute_phi_multiple_partitions(self, hidden_state: torch.Tensor) -> float:
        """
        Compute Phi across multiple possible partitions and return minimum
        (as per IIT definition).
        
        Args:
            hidden_state: Hidden state tensor
            
        Returns:
            Integrated information (minimum across partitions)
        """
        if torch.is_tensor(hidden_state):
            state = hidden_state.detach().cpu().numpy()
        else:
            state = hidden_state
        
        state_flat = state.flatten()
        
        if len(state_flat) < 4:
            return 0.0
        
        state_discrete = self.discretize_state(state_flat)
        
        # Try multiple bipartitions
        phi_values = []
        
        # Equal split
        mid = len(state_discrete) // 2
        part1, part2 = state_discrete[:mid], state_discrete[mid:]
        mi = self.compute_mutual_information(state_discrete, part1, part2)
        phi_values.append(mi)
        
        # 1/3 - 2/3 split
        split = len(state_discrete) // 3
        part1, part2 = state_discrete[:split], state_discrete[split:]
        mi = self.compute_mutual_information(state_discrete, part1, part2)
        phi_values.append(mi)
        
        # 2/3 - 1/3 split
        split = 2 * len(state_discrete) // 3
        part1, part2 = state_discrete[:split], state_discrete[split:]
        mi = self.compute_mutual_information(state_discrete, part1, part2)
        phi_values.append(mi)
        
        # Return minimum (IIT: consciousness = minimum information lost in partition)
        min_phi = min(phi_values)
        
        max_info = np.log2(self.num_states)
        phi_normalized = min_phi / (max_info + 1e-10)
        
        return float(phi_normalized)
    
    def compute_consciousness_score(self,
                                   hidden_states: list,
                                   attention_weights: list) -> Dict[str, float]:
        """
        Compute comprehensive consciousness score.
        
        Args:
            hidden_states: List of hidden states from layers
            attention_weights: List of attention weight tensors
            
        Returns:
            Dictionary of consciousness metrics
        """
        if not hidden_states:
            return {"phi": 0.0, "avg_layer_phi": 0.0}
        
        # Compute Phi for each layer
        phi_values = []
        for state in hidden_states:
            phi = self.compute_phi(state)
            phi_values.append(phi)
        
        # Average Phi across layers
        avg_phi = np.mean(phi_values) if phi_values else 0.0
        
        # Compute integrated attention (how much attention is integrated)
        integrated_attention = 0.0
        if attention_weights:
            for attn in attention_weights:
                if torch.is_tensor(attn):
                    attn = attn.detach().cpu().numpy()
                # Measure how much information is shared across heads
                integrated_attention += np.var(attn)
            integrated_attention /= len(attention_weights)
        
        return {
            "phi_individual": phi_values,
            "avg_layer_phi": float(avg_phi),
            "integrated_attention": float(np.tanh(integrated_attention)),  # Normalize
            "consciousness_index": float((avg_phi + np.tanh(integrated_attention)) / 2),
        }
