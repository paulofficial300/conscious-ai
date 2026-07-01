"""
Multi-Head Attention mechanism for the Conscious Transformer.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with optional consciousness-aware modifications.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 use_bias: bool = True, consciousness_aware: bool = False):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_bias: Whether to use bias in linear layers
            consciousness_aware: Enable consciousness-aligned attention modifications
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.consciousness_aware = consciousness_aware
        
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # For consciousness-aware attention
        if consciousness_aware:
            self.attention_gate = nn.Parameter(torch.ones(num_heads))
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, attention_weights: torch.Tensor = None) -> tuple:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, 1, seq_len_q, seq_len_k) - attention mask
            attention_weights: Optional weights for consciousness modulation
            
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_scores: attention weights for analysis
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply consciousness-aware gating if enabled
        if self.consciousness_aware and attention_weights is not None:
            scores = scores * attention_weights.view(1, self.num_heads, 1, 1)
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function ("gelu", "relu")
        """
        super().__init__()
        
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask
            
        Returns:
            output and attention scores
        """
        # Self-attention with residual connection
        attn_output, attn_scores = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_scores
