#!/usr/bin/env python3
"""
Model Selection and Hyperparameter Optimization for Semantic Embedding Models.

This script performs comprehensive model selection and hyperparameter optimization
for semantic embedding models, exploring different base models and configurations.
"""

import os
import json
import yaml
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import hashlib

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for model selection and hyperparameter optimization."""
    
    # Base models to explore
    base_models: List[str] = None
    
    # Hyperparameter search space
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    epochs: List[int] = None
    temperatures: List[float] = None
    margins: List[float] = None
    negative_sampling_ratios: List[int] = None
    hidden_sizes: List[int] = None
    dropout_rates: List[float] = None
    
    # Optimization settings
    n_trials: int = 50
    timeout: int = 3600  # 1 hour
    n_jobs: int = 1
    
    def __post_init__(self):
        if self.base_models is None:
            self.base_models = [
                "all-MiniLM-L6-v2",                    # Current model (balanced)
                "all-mpnet-base-v2",                   # Best accuracy (standard)
                "paraphrase-multilingual-MiniLM-L12-v2" # Best accuracy (multilingual)
            ]
        
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
        
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64]
        
        if self.epochs is None:
            self.epochs = [5, 10, 15]
        
        if self.temperatures is None:
            self.temperatures = [0.05, 0.1, 0.2]
        
        if self.margins is None:
            self.margins = [0.1, 0.3, 0.5]
        
        if self.negative_sampling_ratios is None:
            self.negative_sampling_ratios = [2, 3, 5]
        
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 384, 512]
        
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3]

@dataclass
class TrialResult:
    """Result from a single optimization trial."""
    
    trial_id: int
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int
    temperature: float
    margin: float
    negative_sampling_ratio: int
    hidden_size: int
    dropout_rate: float
    
    # Training metrics
    train_loss: float
    val_loss: float
    train_time: float
    
    # Evaluation metrics
    similarity_score: float
    contrastive_loss: float
    embedding_quality: float
    
    # Model info
    model_size_mb: float
    inference_time_ms: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ModelSelectionOptimizer:
    """Optimizer for model selection and hyperparameter tuning."""
    
    def __init__(self, config_path: str = "configs/config.yaml", 
                 output_dir: str = "experiments/model_selection"):
        """Initialize the optimizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load optimization config
        self.opt_config = OptimizationConfig()
        
        # Load semantic analysis data
        with open("data/semantic_analysis_data.json", 'r') as f:
            self.semantic_data = json.load(f)
        
        # Results storage
        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
        
        # Setup device
        self.device = self._get_device()
        
        logger.info(f"Initialized optimizer with {len(self.opt_config.base_models)} base models")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _create_model_config(self, trial_params: Dict) -> Dict:
        """Create model configuration from trial parameters."""
        config = self.base_config.copy()
        
        # Update model configuration
        config['model']['encoder']['model_name'] = trial_params['model_name']
        config['model']['encoder']['hidden_size'] = trial_params['hidden_size']
        config['model']['encoder']['dropout'] = trial_params['dropout_rate']
        
        # Update training configuration
        config['training']['batch_size'] = trial_params['batch_size']
        config['model']['training_phases'][0]['epochs'] = trial_params['epochs']
        config['model']['training_phases'][0]['learning_rate'] = trial_params['learning_rate']
        
        # Update contrastive learning configuration
        config['model']['semantic_embedding']['contrastive_learning']['temperature'] = trial_params['temperature']
        config['model']['semantic_embedding']['contrastive_learning']['margin'] = trial_params['margin']
        config['model']['semantic_embedding']['contrastive_learning']['negative_sampling_ratio'] = trial_params['negative_sampling_ratio']
        
        return config
    
    def _train_model(self, trial_params: Dict) -> Tuple[float, float, float]:
        """Train a model with given parameters and return metrics."""
        try:
            # Create temporary config
            temp_config = self._create_model_config(trial_params)
            
            # Save temporary config
            temp_config_path = self.output_dir / f"temp_config_{trial_params['trial_id']}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Import training function
            from models.semantic_embedding_model import train_semantic_embedding_model
            
            # Prepare data
            similar_pairs = self.semantic_data['similar_pairs']
            sentences = [signal['sentence'] for signal in self.semantic_data['training_signals']]
            
            # Train model
            start_time = time.time()
            model = train_semantic_embedding_model(
                similar_pairs=similar_pairs,
                sentences=sentences,
                config=temp_config,
                device=self.device
            )
            train_time = time.time() - start_time
            
            # Evaluate model
            val_loss = self._evaluate_model(model, similar_pairs[:100])  # Use subset for validation
            train_loss = self._evaluate_model(model, similar_pairs[:50])  # Use subset for training loss
            
            # Clean up
            temp_config_path.unlink(missing_ok=True)
            
            return train_loss, val_loss, train_time
            
        except Exception as e:
            logger.error(f"Training failed for trial {trial_params['trial_id']}: {e}")
            return float('inf'), float('inf'), 0.0
    
    def _evaluate_model(self, model, similar_pairs: List[Dict]) -> float:
        """Evaluate model performance on similar pairs."""
        try:
            model.eval()
            total_loss = 0.0
            
            with torch.no_grad():
                for pair in similar_pairs:
                    # Create positive pair
                    sent1, sent2 = pair['sentence1'], pair['sentence2']
                    
                    # Forward pass
                    outputs = model([sent1], [sent2])
                    embeddings1 = outputs['embeddings1']
                    embeddings2 = outputs['embeddings2']
                    
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
                    
                    # Loss based on similarity (should be high for similar pairs)
                    loss = -torch.log(torch.sigmoid(similarity) + 1e-8)
                    total_loss += loss.item()
            
            return total_loss / len(similar_pairs)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return float('inf')
    
    def _compute_model_metrics(self, model, trial_params: Dict) -> Dict:
        """Compute additional model metrics."""
        try:
            # Model size
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            # Inference time
            test_sentences = ["This is a test sentence."] * 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_sentences, test_sentences)
            
            inference_time_ms = (time.time() - start_time) * 1000 / 100  # Average per sentence
            
            return {
                'model_size_mb': model_size_mb,
                'inference_time_ms': inference_time_ms
            }
            
        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            return {'model_size_mb': 0.0, 'inference_time_ms': 0.0}
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Suggest parameters
        params = {
            'trial_id': trial.number,
            'model_name': trial.suggest_categorical('model_name', self.opt_config.base_models),
            'learning_rate': trial.suggest_categorical('learning_rate', self.opt_config.learning_rates),
            'batch_size': trial.suggest_categorical('batch_size', self.opt_config.batch_sizes),
            'epochs': trial.suggest_categorical('epochs', self.opt_config.epochs),
            'temperature': trial.suggest_categorical('temperature', self.opt_config.temperatures),
            'margin': trial.suggest_categorical('margin', self.opt_config.margins),
            'negative_sampling_ratio': trial.suggest_categorical('negative_sampling_ratio', self.opt_config.negative_sampling_ratios),
            'hidden_size': trial.suggest_categorical('hidden_size', self.opt_config.hidden_sizes),
            'dropout_rate': trial.suggest_categorical('dropout_rate', self.opt_config.dropout_rates)
        }
        
        # Train model
        train_loss, val_loss, train_time = self._train_model(params)
        
        # Create temporary model for metrics
        temp_config = self._create_model_config(params)
        from models.semantic_embedding_model import SemanticEmbeddingModel
        temp_model = SemanticEmbeddingModel(
            model_name=params['model_name'],
            embedding_dim=params['hidden_size']
        )
        
        # Compute additional metrics
        metrics = self._compute_model_metrics(temp_model, params)
        
        # Create result
        result = TrialResult(
            trial_id=params['trial_id'],
            model_name=params['model_name'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            temperature=params['temperature'],
            margin=params['margin'],
            negative_sampling_ratio=params['negative_sampling_ratio'],
            hidden_size=params['hidden_size'],
            dropout_rate=params['dropout_rate'],
            train_loss=train_loss,
            val_loss=val_loss,
            train_time=train_time,
            similarity_score=1.0 / (1.0 + val_loss),  # Convert loss to similarity score
            contrastive_loss=val_loss,
            embedding_quality=1.0 / (1.0 + val_loss),
            model_size_mb=metrics['model_size_mb'],
            inference_time_ms=metrics['inference_time_ms']
        )
        
        self.results.append(result)
        
        # Update best result
        if self.best_result is None or result.val_loss < self.best_result.val_loss:
            self.best_result = result
        
        return val_loss
    
    def run_optimization(self) -> Dict:
        """Run the complete optimization process."""
        logger.info("Starting model selection and hyperparameter optimization...")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs
        )
        
        # Save results
        self._save_results(study)
        
        # Create visualizations
        self._create_visualizations()
        
        logger.info(f"Optimization completed! Best validation loss: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }
    
    def _save_results(self, study: optuna.Study):
        """Save optimization results."""
        # Save all results
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        results_df.to_csv(self.output_dir / "optimization_results.csv", index=False)
        
        # Save best result
        if self.best_result:
            with open(self.output_dir / "best_result.json", 'w') as f:
                json.dump(self.best_result.to_dict(), f, indent=2)
        
        # Save study
        with open(self.output_dir / "study.pkl", 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        # Save summary
        summary = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(self.results),
            'optimization_time': time.time(),
            'best_result': self.best_result.to_dict() if self.best_result else None
        }
        
        with open(self.output_dir / "optimization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_visualizations(self):
        """Create visualizations of optimization results."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create results DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Selection and Hyperparameter Optimization Results', fontsize=16)
        
        # 1. Validation loss by model
        axes[0, 0].boxplot([df[df['model_name'] == model]['val_loss'] for model in df['model_name'].unique()])
        axes[0, 0].set_title('Validation Loss by Model')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].set_xticklabels(df['model_name'].unique(), rotation=45)
        
        # 2. Learning rate vs validation loss
        axes[0, 1].scatter(df['learning_rate'], df['val_loss'], alpha=0.6)
        axes[0, 1].set_title('Learning Rate vs Validation Loss')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_xscale('log')
        
        # 3. Batch size vs validation loss
        axes[0, 2].boxplot([df[df['batch_size'] == bs]['val_loss'] for bs in df['batch_size'].unique()])
        axes[0, 2].set_title('Validation Loss by Batch Size')
        axes[0, 2].set_ylabel('Validation Loss')
        axes[0, 2].set_xticklabels(df['batch_size'].unique())
        
        # 4. Training time vs validation loss
        axes[1, 0].scatter(df['train_time'], df['val_loss'], alpha=0.6)
        axes[1, 0].set_title('Training Time vs Validation Loss')
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_ylabel('Validation Loss')
        
        # 5. Model size vs validation loss
        axes[1, 1].scatter(df['model_size_mb'], df['val_loss'], alpha=0.6)
        axes[1, 1].set_title('Model Size vs Validation Loss')
        axes[1, 1].set_xlabel('Model Size (MB)')
        axes[1, 1].set_ylabel('Validation Loss')
        
        # 6. Inference time vs validation loss
        axes[1, 2].scatter(df['inference_time_ms'], df['val_loss'], alpha=0.6)
        axes[1, 2].set_title('Inference Time vs Validation Loss')
        axes[1, 2].set_xlabel('Inference Time (ms)')
        axes[1, 2].set_ylabel('Validation Loss')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimization_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")

def main():
    """Main function for model selection and hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Model selection and hyperparameter optimization")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="experiments/model_selection",
                       help="Output directory for results")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Optimization timeout in seconds")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelSelectionOptimizer(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Update optimization config
    optimizer.opt_config.n_trials = args.n_trials
    optimizer.opt_config.timeout = args.timeout
    
    # Run optimization
    results = optimizer.run_optimization()
    
    print(f"\n=== OPTIMIZATION COMPLETED ===")
    print(f"Best validation loss: {results['best_value']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Total trials: {results['n_trials']}")
    print(f"Results saved to: {args.output_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 