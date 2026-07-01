"""
Main training loop for Conscious Transformer.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


class ConsciousTransformerTrainer:
    """
    Trainer for Conscious Transformer with consciousness monitoring.
    """
    
    def __init__(self,
                 model,
                 gwt_hub,
                 hot_module,
                 consciousness_metrics,
                 loss_fn,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: str = "models/checkpoints"):
        """
        Args:
            model: ConsciousTransformer instance
            gwt_hub: WorkspaceHub instance
            hot_module: HigherOrderThought instance
            consciousness_metrics: ConsciousnessMetrics instance
            loss_fn: Loss function
            optimizer: Optimizer (creates default if None)
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.gwt_hub = gwt_hub.to(device) if gwt_hub is not None else None
        self.hot_module = hot_module.to(device) if hot_module is not None else None
        self.consciousness_metrics = consciousness_metrics
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Setup optimizer
        params = list(model.parameters())
        if gwt_hub is not None:
            params.extend(gwt_hub.parameters())
        if hot_module is not None:
            params.extend(hot_module.parameters())
        
        self.optimizer = optimizer or optim.AdamW(params, lr=5e-4, weight_decay=0.01)
        
        # Training statistics
        self.global_step = 0
        self.train_history = {
            "steps": [],
            "losses": [],
            "consciousness_metrics": [],
        }
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: (input_ids, target_ids)
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        if self.gwt_hub is not None:
            self.gwt_hub.train()
        if self.hot_module is not None:
            self.hot_module.train()
        
        input_ids, targets = batch
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass through transformer
        transformer_out = self.model(input_ids, return_attention=True)
        logits = transformer_out["logits"]
        hidden_states = transformer_out["hidden_states"]
        attention_scores = transformer_out["attention_scores"]
        
        # HOT evaluation
        hot_out = None
        if self.hot_module is not None:
            hot_out = self.hot_module(hidden_states[-1])
            hot_signal = hot_out["hot_signal"]
        else:
            hot_signal = None
        
        # GWT processing
        gwt_out = None
        if self.gwt_hub is not None:
            gwt_out = self.gwt_hub(hidden_states[-1], hot_signal)
            specialist_activity = self.gwt_hub.get_specialist_activity()
        else:
            specialist_activity = None
        
        # Compute consciousness metrics
        self.consciousness_metrics.update(
            hidden_states=hidden_states,
            attention_scores=attention_scores,
            specialist_activity=specialist_activity,
            evaluation_scores=hot_out["evaluation"] if hot_out else None,
            competition_scores=gwt_out["competition_scores"] if gwt_out else None,
        )
        
        consciousness_m = self.consciousness_metrics.get_current_metrics()
        workspace_util = consciousness_m.get("workspace_utilization_current", 0.5)
        hot_conf = hot_out["evaluation"]["confidence"] if hot_out else None
        
        # Compute loss
        loss_dict = self.loss_fn(
            logits,
            targets,
            consciousness_metrics=consciousness_m,
            workspace_utilization=workspace_util,
            hot_confidence=hot_conf,
        )
        
        total_loss = loss_dict["total_loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Prepare output
        output = {}
        for key, val in loss_dict.items():
            if isinstance(val, torch.Tensor):
                output[key] = val.detach().item()
            else:
                output[key] = float(val)
        
        output.update(consciousness_m)
        
        self.global_step += 1
        
        return output
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation pass.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        if self.gwt_hub is not None:
            self.gwt_hub.eval()
        if self.hot_module is not None:
            self.hot_module.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            transformer_out = self.model(input_ids, return_attention=False)
            logits = transformer_out["logits"]
            
            loss_dict = self.loss_fn(logits, targets)
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }
    
    def train_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader] = None,
                   log_frequency: int = 10) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            log_frequency: Log every N steps
            
        Returns:
            Epoch statistics
        """
        epoch_losses = []
        
        for step, batch in enumerate(train_loader):
            metrics = self.train_step(batch)
            epoch_losses.append(metrics.get("total_loss", 0.0))
            
            if (step + 1) % log_frequency == 0:
                avg_loss = sum(epoch_losses[-log_frequency:]) / log_frequency
                print(f"Step {self.global_step}, Loss: {avg_loss:.4f}")
            
            self.train_history["steps"].append(self.global_step)
            self.train_history["losses"].append(metrics.get("total_loss", 0.0))
            self.train_history["consciousness_metrics"].append(metrics)
        
        result = {
            "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
        }
        
        if val_loader is not None:
            val_metrics = self.validate(val_loader)
            result.update(val_metrics)
        
        return result
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "gwt_state": self.gwt_hub.state_dict() if self.gwt_hub else None,
            "hot_state": self.hot_module.state_dict() if self.hot_module else None,
            "optimizer_state": self.optimizer.state_dict(),
            "train_history": self.train_history,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.gwt_hub and checkpoint.get("gwt_state"):
            self.gwt_hub.load_state_dict(checkpoint["gwt_state"])
        if self.hot_module and checkpoint.get("hot_state"):
            self.hot_module.load_state_dict(checkpoint["hot_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = checkpoint["global_step"]
        self.train_history = checkpoint["train_history"]
        print(f"Checkpoint loaded from {path}")
