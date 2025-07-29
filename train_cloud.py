 #!/usr/bin/env python3
"""
Cloud-optimized training script for book sentence classification.
Supports GPU training, checkpointing, and cloud-specific features.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW
import yaml
import json
from datasets import load_from_disk
import wandb
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.constructive_model import ConstructiveLearningModel
from utils.data_utils import create_dataloaders, tokenize_dataset

class CloudTrainingManager:
    """Cloud-optimized training manager with GPU support."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize cloud training manager."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb for experiment tracking
        self.setup_wandb()
        
        # Load data and model
        self.load_data()
        self.load_model()
        
    def setup_logging(self):
        """Setup logging for cloud training."""
        log_dir = Path("experiments/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "cloud_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_wandb(self):
        """Setup Weights & Biases for experiment tracking."""
        try:
            wandb.init(
                project="books-classification",
                config=self.config,
                name=f"cloud-training-{self.config.get('model', {}).get('name', 'constructive')}"
            )
            self.logger.info("WandB initialized successfully")
        except Exception as e:
            self.logger.warning(f"WandB initialization failed: {e}")
            wandb = None
            
    def load_data(self):
        """Load and prepare data for cloud training."""
        self.logger.info("Loading dataset...")
        
        # Load processed dataset
        dataset_path = Path("data/processed_dataset")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run data preparation first.")
            
        self.dataset = load_from_disk(str(dataset_path))
        self.logger.info(f"Dataset loaded: {len(self.dataset['train'])} train, {len(self.dataset['validation'])} val")
        
        # Load tokenizer
        model_name = self.config['model']['encoder_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize datasets
        self.tokenized_datasets = {}
        for split in ['train', 'validation', 'test']:
            self.tokenized_datasets[split] = tokenize_dataset(
                self.dataset[split],
                self.tokenizer,
                max_length=self.config['data']['max_sentence_length']
            )
            
        self.logger.info("Data preparation completed")
        
    def load_model(self):
        """Load model for cloud training."""
        self.logger.info("Loading model...")
        
        self.model = ConstructiveLearningModel()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")
        
    def create_dataloaders(self):
        """Create dataloaders optimized for cloud training."""
        batch_size = self.config['training']['batch_size']
        
        # Adjust batch size for GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:  # Less than 8GB
                batch_size = min(batch_size, 8)
            elif gpu_memory < 16:  # Less than 16GB
                batch_size = min(batch_size, 16)
            else:
                batch_size = min(batch_size, 32)
                
        self.logger.info(f"Using batch size: {batch_size}")
        
        self.dataloaders = {}
        for split in ['train', 'validation', 'test']:
            self.dataloaders[split] = DataLoader(
                self.tokenized_datasets[split],
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=min(4, os.cpu_count()),  # Limit workers for cloud
                pin_memory=torch.cuda.is_available()
            )
            
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_acc': 100 * correct / total,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
                
        return total_loss / len(self.dataloaders['train']), 100 * correct / total
        
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['validation'], desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss / len(self.dataloaders['validation']), 100 * correct / total
        
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint_dir = Path("experiments/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
            
    def train(self, num_epochs: int = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
            
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Create dataloaders
        self.create_dataloaders()
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        total_steps = len(self.dataloaders['train']) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log results
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss_epoch': train_loss,
                    'train_acc_epoch': train_acc
                })
                
        self.logger.info("Training completed!")
        
    def cleanup(self):
        """Cleanup resources."""
        if wandb.run is not None:
            wandb.finish()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cloud training for book classification")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Initialize training manager
    trainer = CloudTrainingManager(args.config)
    
    try:
        # Start training
        trainer.train(args.epochs)
    finally:
        # Cleanup
        trainer.cleanup()

if __name__ == "__main__":
    main()