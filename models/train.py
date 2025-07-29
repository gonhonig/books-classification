"""
Training script for constructive and self-supervised learning model.
"""

import os
import sys
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from datasets import DatasetDict, load_from_disk
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constructive_model import ConstructiveLearningModel, ConstructiveTrainer
from utils.data_utils import create_dataloaders, tokenize_dataset
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_curves

class TrainingManager:
    """Manages the complete training process."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        # Create directories
        self._create_directories()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['logging']
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
        
    def _setup_experiment_tracking(self):
        """Setup experiment tracking with wandb."""
        if self.config['logging']['wandb_entity']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                config=self.config
            )
            
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['training']['checkpoint_dir'],
            self.config['training']['log_dir'],
            "experiments/results",
            "experiments/visualizations"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def load_data(self):
        """Load and prepare data."""
        self.logger.info("Loading data...")
        
        # Load processed dataset
        dataset_path = Path("data/processed_dataset")
        if dataset_path.exists():
            self.dataset = load_from_disk(str(dataset_path))
        else:
            raise FileNotFoundError("Processed dataset not found. Run data/prepare_data.py first.")
            
        # Load metadata
        metadata_path = Path("data/metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError("Metadata not found.")
            
        # Load self-supervised datasets
        ss_dataset_path = Path("data/self_supervised_dataset")
        if ss_dataset_path.exists():
            self.ss_dataset = load_from_disk(str(ss_dataset_path))
        else:
            self.logger.warning("Self-supervised datasets not found. Creating them...")
            self.ss_dataset = None
            
        self.logger.info(f"Loaded dataset with {len(self.dataset['train'])} training samples")
        
    def prepare_dataloaders(self):
        """Prepare dataloaders for all phases."""
        self.logger.info("Preparing dataloaders...")
        
        # Tokenize datasets
        tokenizer = ConstructiveLearningModel().tokenizer
        
        # Main classification dataset
        tokenized_dataset = tokenize_dataset(self.dataset, tokenizer, 
                                          max_length=self.config['data']['max_sentence_length'])
        
        # Create dataloaders for main dataset
        self.main_dataloaders = create_dataloaders(
            tokenized_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware']['num_workers']
        )
        
        # Create dataloaders for self-supervised tasks
        if self.ss_dataset:
            self.ss_dataloaders = {}
            
            # MLM dataloader
            mlm_tokenized = tokenize_dataset(self.ss_dataset['mlm'], tokenizer, 
                                           max_length=self.config['data']['max_sentence_length'],
                                           task='mlm')
            self.ss_dataloaders['mlm'] = create_dataloaders(
                mlm_tokenized,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['hardware']['num_workers']
            )
            
            # NSP dataloader
            nsp_tokenized = tokenize_dataset(self.ss_dataset['nsp'], tokenizer,
                                           max_length=self.config['data']['max_sentence_length'],
                                           task='nsp')
            self.ss_dataloaders['nsp'] = create_dataloaders(
                nsp_tokenized,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['hardware']['num_workers']
            )
            
            # Similarity dataloader
            similarity_tokenized = tokenize_dataset(self.ss_dataset['similarity'], tokenizer,
                                                 max_length=self.config['data']['max_sentence_length'],
                                                 task='similarity')
            self.ss_dataloaders['similarity'] = create_dataloaders(
                similarity_tokenized,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['hardware']['num_workers']
            )
            
        self.logger.info("Dataloaders prepared successfully")
        
    def create_model_and_trainer(self):
        """Create model and trainer."""
        self.logger.info("Creating model and trainer...")
        
        self.model = ConstructiveLearningModel()
        self.trainer = ConstructiveTrainer(self.model)
        
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def create_optimizers_and_schedulers(self):
        """Create optimizers and schedulers for all phases."""
        self.logger.info("Creating optimizers and schedulers...")
        
        constructive_config = self.config['model']['constructive_learning']
        
        self.optimizers = []
        self.schedulers = []
        
        for phase_config in constructive_config['phases']:
            # Create optimizer
            optimizer = AdamW(
                self.model.parameters(),
                lr=phase_config['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            self.optimizers.append(optimizer)
            
            # Create scheduler
            total_steps = len(self.main_dataloaders['train']) * phase_config['epochs']
            warmup_steps = self.config['training']['warmup_steps']
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            self.schedulers.append(scheduler)
            
        self.logger.info(f"Created {len(self.optimizers)} optimizers and schedulers")
        
    def train_constructive_phases(self):
        """Train all constructive learning phases."""
        self.logger.info("Starting constructive learning training...")
        
        constructive_config = self.config['model']['constructive_learning']
        phase_results = []
        
        for phase_idx, phase_config in enumerate(constructive_config['phases']):
            self.logger.info(f"Training phase {phase_idx}: {phase_config['name']}")
            
            # Prepare dataloaders for this phase
            phase_dataloaders = self._prepare_phase_dataloaders(phase_idx, phase_config)
            
            # Train phase
            phase_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(phase_config['epochs']):
                # Training
                train_loss = self._train_epoch(phase_idx, phase_dataloaders['train'], 
                                             self.optimizers[phase_idx], self.schedulers[phase_idx])
                
                # Validation
                val_loss = self._validate_epoch(phase_idx, phase_dataloaders['validation'])
                
                # Logging
                self.logger.info(f"Phase {phase_idx}, Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if wandb.run:
                    wandb.log({
                        f"phase_{phase_idx}_train_loss": train_loss,
                        f"phase_{phase_idx}_val_loss": val_loss,
                        f"phase_{phase_idx}_epoch": epoch
                    })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model for this phase
                    self._save_checkpoint(phase_idx, epoch, val_loss)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info(f"Early stopping triggered for phase {phase_idx}")
                    break
                    
                phase_losses.append({'train': train_loss, 'val': val_loss})
                
            # Save phase knowledge
            self.model.save_phase_knowledge(phase_idx, 
                                          f"experiments/checkpoints/phase_{phase_idx}_knowledge.pt")
            
            phase_results.append({
                'phase': phase_idx,
                'name': phase_config['name'],
                'losses': phase_losses,
                'best_val_loss': best_val_loss
            })
            
        return phase_results
        
    def _prepare_phase_dataloaders(self, phase_idx: int, phase_config: dict):
        """Prepare dataloaders for a specific phase."""
        tasks = phase_config['tasks']
        
        if 'all' in tasks:
            # Use all dataloaders
            return {
                'train': [self.main_dataloaders['train']] + list(self.ss_dataloaders.values()),
                'validation': self.main_dataloaders['validation'],
                'test': self.main_dataloaders['test']
            }
        else:
            # Use specific dataloaders based on tasks
            train_dataloaders = [self.main_dataloaders['train']]
            
            for task in tasks:
                if task in self.ss_dataloaders:
                    train_dataloaders.append(self.ss_dataloaders[task]['train'])
                    
            return {
                'train': train_dataloaders,
                'validation': self.main_dataloaders['validation'],
                'test': self.main_dataloaders['test']
            }
            
    def _train_epoch(self, phase_idx: int, dataloaders, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Combine all dataloaders if multiple
        if isinstance(dataloaders, list):
            all_batches = []
            for dataloader in dataloaders:
                all_batches.extend(list(dataloader))
        else:
            all_batches = list(dataloaders)
            
        for batch in tqdm(all_batches, desc=f"Phase {phase_idx} Training"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task='all'  # Use all tasks for training
            )
            
            # Compute loss
            loss = self.model.compute_loss(outputs, **batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            
            optimizer.step()
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def _validate_epoch(self, phase_idx: int, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Phase {phase_idx} Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task='classification'
                )
                
                # Compute loss
                loss = self.model.compute_loss(outputs, **batch)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def _save_checkpoint(self, phase_idx: int, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = f"experiments/checkpoints/phase_{phase_idx}_epoch_{epoch}.pt"
        
        torch.save({
            'phase': phase_idx,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizers[phase_idx].state_dict(),
            'scheduler_state_dict': self.schedulers[phase_idx].state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def evaluate_final_model(self):
        """Evaluate the final trained model."""
        self.logger.info("Evaluating final model...")
        
        # Load best model from last phase
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            self._load_checkpoint(best_checkpoint)
            
        # Evaluate on test set
        test_metrics = evaluate_model(
            self.model,
            self.main_dataloaders['test'],
            self.device,
            self.metadata
        )
        
        # Save results
        results_path = "experiments/results/final_evaluation.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
            
        self.logger.info(f"Final evaluation results saved to {results_path}")
        
        return test_metrics
        
    def _find_best_checkpoint(self):
        """Find the best checkpoint based on validation loss."""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return None
            
        # Find checkpoint with lowest validation loss
        best_checkpoint = None
        best_val_loss = float('inf')
        
        for checkpoint in checkpoints:
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            val_loss = checkpoint_data.get('val_loss', float('inf'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = checkpoint
                
        return best_checkpoint
        
    def _load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
    def run_training(self):
        """Run the complete training process."""
        self.logger.info("Starting training process...")
        
        # Load data
        self.load_data()
        
        # Prepare dataloaders
        self.prepare_dataloaders()
        
        # Create model and trainer
        self.create_model_and_trainer()
        
        # Create optimizers and schedulers
        self.create_optimizers_and_schedulers()
        
        # Train constructive phases
        phase_results = self.train_constructive_phases()
        
        # Evaluate final model
        final_metrics = self.evaluate_final_model()
        
        # Plot training curves
        plot_training_curves(phase_results, save_path="experiments/visualizations/training_curves.png")
        
        self.logger.info("Training completed successfully!")
        
        return phase_results, final_metrics

def main():
    """Main training function."""
    # Set random seed
    import yaml
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    
    # Create training manager
    manager = TrainingManager()
    
    # Run training
    phase_results, final_metrics = manager.run_training()
    
    print("Training completed!")
    print(f"Final metrics: {final_metrics}")

if __name__ == "__main__":
    main() 