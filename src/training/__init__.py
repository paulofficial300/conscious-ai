"""
Training module with integrated consciousness monitoring.
"""

from .trainer import ConsciousTransformerTrainer
from .losses import ConsciousnessLoss

__all__ = [
    "ConsciousTransformerTrainer",
    "ConsciousnessLoss",
]
