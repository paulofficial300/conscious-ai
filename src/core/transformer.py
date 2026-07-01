"""
Core Conscious Transformer Architecture.
Combines transformer layers with consciousness-inspired mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .attention import TransformerLayer
from .embeddings import TokenEmbedding, PositionalEncoding


class ConsciousTransformer(nn.Module):
    """
    Transformer-based model with consciousness-inspired architecture.
    Designed to work with Global Workspace Theory and Higher-Order Thought modules.
    """
    
    def __init__(self, 
                 vocab_size: int = 50257,
                 d_model: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 padding_idx: int = 0,
                 consciousness_enabled: bool = True):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            activation: Activation function
            padding_idx: Padding token index
            consciousness_enabled: Enable consciousness mechanisms
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.consciousness_enabled = consciousness_enabled
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, method="sinusoidal")
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Consciousness-related components
        if consciousness_enabled:
            # Store for consciousness metrics
            self.register_buffer("_attention_history", 
                               torch.zeros(num_layers, num_heads))
            self.register_buffer("_layer_activity", 
                               torch.zeros(num_layers))
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_attention: Whether to return attention scores
            
        Returns:
            Dictionary containing:
                - logits: (batch_size, seq_len, vocab_size)
                - hidden_states: List of hidden states from each layer
                - attention_scores: Attention weights if return_attention=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Store hidden states for consciousness analysis
        hidden_states = [x]
        all_attention_scores = []
        
        # Forward pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x, attn_scores = layer(x, attention_mask)
            hidden_states.append(x)
            all_attention_scores.append(attn_scores)
            
            # Update consciousness metrics
            if self.consciousness_enabled:
                self._layer_activity[i] = attn_scores.mean().detach()
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_layer(x)
        
        # Prepare output dictionary
        output = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if return_attention:
            output["attention_scores"] = all_attention_scores
        
        return output
    
    def get_consciousness_metrics(self) -> Dict[str, float]:
        """Get consciousness-related metrics."""
        if not self.consciousness_enabled:
            return {}
        
        return {
            "avg_layer_activity": self._layer_activity.mean().item(),
            "max_layer_activity": self._layer_activity.max().item(),
            "min_layer_activity": self._layer_activity.min().item(),
        }
    
    def reset_consciousness_metrics(self):
        """Reset consciousness metrics."""
        if self.consciousness_enabled:
            self._layer_activity.fill_(0)
