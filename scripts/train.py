"""
Main training script for Conscious Transformer.
"""

import argparse
import torch
from pathlib import Path
import yaml

# Import components
from src.core.transformer import ConsciousTransformer
from src.gwt.workspace_hub import WorkspaceHub
from src.hot.hot_module import HigherOrderThought
from src.consciousness.metrics import ConsciousnessMetrics
from src.training.losses import ConsciousnessLoss
from src.training.trainer import ConsciousTransformerTrainer


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dummy_dataloader(batch_size: int = 32, seq_len: int = 256, 
                           num_batches: int = 100, vocab_size: int = 50257):
    """
    Create dummy data loader for testing.
    In production, this would load real data.
    """
    import random
    
    class DummyDataset:
        def __init__(self, num_batches, batch_size, seq_len, vocab_size):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.num_batches
        
        def __getitem__(self, idx):
            # Return random token sequences
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            return input_ids, targets
    
    dataset = DummyDataset(num_batches, batch_size, seq_len, vocab_size)
    return torch.utils.data.DataLoader(dataset, batch_size=1)


def main(args):
    """Main training function."""
    
    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    gwt_config = load_config(args.gwt_config)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model components
    print("Initializing Conscious Transformer...")
    transformer = ConsciousTransformer(
        vocab_size=model_config["model"].get("vocab_size", 50257),
        d_model=model_config["model"].get("d_model", 768),
        num_layers=model_config["model"].get("num_layers", 12),
        num_heads=model_config["model"].get("num_heads", 12),
        d_ff=model_config["model"].get("d_ff", 3072),
        max_seq_length=model_config["model"].get("max_seq_length", 1024),
        dropout=model_config["model"].get("dropout", 0.1),
        consciousness_enabled=model_config["consciousness"].get("enabled", True),
    )
    
    print("Initializing Global Workspace Theory Hub...")
    gwt_hub = WorkspaceHub(
        transformer_dim=model_config["model"]["d_model"],
        workspace_dim=gwt_config["gwt_workspace"]["workspace_dimension"],
        workspace_capacity=gwt_config["gwt_workspace"]["workspace_capacity"],
        num_specialists=len(gwt_config["gwt_workspace"]["specialists"]),
        specialist_configs=gwt_config["gwt_workspace"]["specialists"],
    )
    
    print("Initializing Higher-Order Thought Module...")
    hot_module = HigherOrderThought(
        input_dim=model_config["model"]["d_model"],
        evaluation_layers=model_config["hot"].get("evaluation_layers", 2),
        evaluation_hidden_dim=model_config["hot"].get("evaluation_hidden_dim", 256),
        confidence_threshold=model_config["hot"].get("confidence_threshold", 0.5),
        metacognitive_depth=model_config["hot"].get("metacognitive_depth", 3),
    )
    
    print("Initializing Consciousness Metrics...")
    consciousness_metrics = ConsciousnessMetrics()
    
    print("Initializing Loss Function...")
    loss_fn = ConsciousnessLoss(
        language_weight=1.0,
        consciousness_weight=0.1,
        workspace_efficiency_weight=0.05,
        hot_confidence_weight=0.05,
    )
    
    print("Initializing Trainer...")
    trainer = ConsciousTransformerTrainer(
        model=transformer,
        gwt_hub=gwt_hub,
        hot_module=hot_module,
        consciousness_metrics=consciousness_metrics,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create data loaders
    print("Loading data...")
    batch_size = training_config["training"].get("batch_size", 32)
    train_loader = create_dummy_dataloader(
        batch_size=batch_size,
        seq_len=model_config["model"].get("max_seq_length", 256),
        num_batches=training_config["training"].get("num_epochs", 100),
        vocab_size=model_config["model"].get("vocab_size", 50257),
    )
    
    val_loader = create_dummy_dataloader(
        batch_size=batch_size,
        seq_len=model_config["model"].get("max_seq_length", 256),
        num_batches=10,
        vocab_size=model_config["model"].get("vocab_size", 50257),
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    num_epochs = training_config["training"].get("num_epochs", 10)
    log_frequency = training_config["logging"].get("log_frequency", 10)
    save_frequency = training_config["logging"].get("save_frequency", 500)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        epoch_stats = trainer.train_epoch(
            train_loader,
            val_loader,
            log_frequency=log_frequency,
        )
        
        print(f"Train Loss: {epoch_stats.get('train_loss', 0):.4f}")
        if 'val_loss' in epoch_stats:
            print(f"Val Loss: {epoch_stats.get('val_loss', 0):.4f}")
            print(f"Val Perplexity: {epoch_stats.get('val_perplexity', 0):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path, epoch)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conscious Transformer")
    parser.add_argument("--model_config", default="configs/model_config.yaml",
                       help="Path to model config")
    parser.add_argument("--training_config", default="configs/training_config.yaml",
                       help="Path to training config")
    parser.add_argument("--gwt_config", default="configs/gwt_config.yaml",
                       help="Path to GWT config")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints",
                       help="Directory for checkpoints")
    
    args = parser.parse_args()
    main(args)
