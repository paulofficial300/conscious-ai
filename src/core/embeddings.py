"""
Token and Positional Embedding layers for the Conscious Transformer.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding layer that converts token IDs to dense vectors."""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding
            padding_idx: Index for padding tokens
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    Alternative: learnable positional embeddings.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 2048, method: str = "sinusoidal"):
        """
        Args:
            d_model: Dimension of the encoding
            max_seq_length: Maximum sequence length to encode
            method: "sinusoidal" or "learnable"
        """
        super().__init__()
        self.d_model = d_model
        self.method = method
        
        if method == "sinusoidal":
            self._init_sinusoidal(max_seq_length)
        elif method == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        else:
            raise ValueError(f"Unknown positional encoding method: {method}")
    
    def _init_sinusoidal(self, max_seq_length: int):
        """Initialize sinusoidal positional encoding."""
        pe = torch.zeros(max_seq_length, self.d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        
        if self.method == "sinusoidal":
            pos_enc = self.pe[:, :seq_len, :]
        else:  # learnable
            pos_enc = self.pos_embedding[:, :seq_len, :]
        
        return x + pos_enc
