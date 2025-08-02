"""
Hyperparameter Optimization for Individual Book Models
Optimizes hyperparameters for each book model using grid search and cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import json
from datetime import datetime
import itertools
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryBookClassifier(nn.Module):
    """Binary classifier for a single book."""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(BinaryBookClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim
        
        # Output layer for binary classification
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Binary classification output
        x = self.output_layer(x)
        return x.squeeze()

class HyperparameterOptimizer:
    """Optimize hyperparameters for individual book models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Book names mapping
        self.book_names = {
            'book_1': 'Anna Karenina',
            'book_2': 'Wuthering Heights', 
            'book_3': 'Frankenstein',
            'book_4': 'The Adventures of Alice in Wonderland'
        }
        
        # Load the semantic augmented dataset
        self.df = pd.read_csv('data/semantic_augmented/semantic_augmented_dataset.csv')
        self.embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')['embeddings'].astype(np.float32)
        
        # Extract book labels
        column_mapping = {
            'book_Anna_Karenina': 'book_1',
            'book_Wuthering_Heights': 'book_2', 
            'book_Frankenstein': 'book_3',
            'book_The_Adventures_of_Alice_in_Wonderland': 'book_4'
        }
        
        self.book_labels = {}
        for col_name, book_col in column_mapping.items():
            if col_name in self.df.columns:
                self.book_labels[book_col] = self.df[col_name].values.astype(np.float32)
        
        logger.info(f"Loaded dataset with {len(self.df)} sentences")
        
    def create_balanced_dataset(self, book_col, target_samples_per_class=5000):
        """Create a balanced dataset for a specific book while preserving multi-label sentences."""
        logger.info(f"Creating balanced dataset for {self.book_names[book_col]}")
        
        # Get positive and negative samples
        positive_indices = np.where(self.book_labels[book_col] == 1)[0]
        negative_indices = np.where(self.book_labels[book_col] == 0)[0]
        
        # Identify multi-label sentences
        multi_label_indices = []
        for i in range(len(self.embeddings)):
            book_labels_for_sentence = [
                self.book_labels['book_1'][i],
                self.book_labels['book_2'][i], 
                self.book_labels['book_3'][i],
                self.book_labels['book_4'][i]
            ]
            if sum(book_labels_for_sentence) > 1:
                multi_label_indices.append(i)
        
        # For positive samples, prioritize multi-label sentences
        positive_multi_label = [i for i in positive_indices if i in multi_label_indices]
        positive_single_label = [i for i in positive_indices if i not in multi_label_indices]
        
        # Include ALL multi-label sentences that are positive for this book
        selected_positive = positive_multi_label.copy()
        
        # Add single-label positive samples to reach target
        remaining_positive_needed = target_samples_per_class - len(selected_positive)
        if remaining_positive_needed > 0 and len(positive_single_label) > 0:
            n_to_sample = min(remaining_positive_needed, len(positive_single_label))
            selected_single_label = np.random.choice(positive_single_label, n_to_sample, replace=False)
            selected_positive.extend(selected_single_label)
        
        # For negative samples, sample normally
        n_negative = min(len(negative_indices), len(selected_positive))
        selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
        
        # Combine indices
        original_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(original_indices)
        
        # Create balanced dataset
        X_balanced = self.embeddings[original_indices]
        y_balanced = self.book_labels[book_col][original_indices]
        
        return X_balanced, y_balanced, original_indices
    
    def train_model_with_hyperparameters(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train a model with specific hyperparameters and return validation metrics."""
        # Create model
        model = BinaryBookClassifier(
            input_dim=384,
            hidden_dims=hyperparams['hidden_dims'],
            dropout_rate=hyperparams['dropout_rate']
        ).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(hyperparams['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= hyperparams['patience']:
                break
        
        # Calculate final validation metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_pred_binary = (val_predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(val_targets, val_pred_binary)
        precision = precision_score(val_targets, val_pred_binary)
        recall = recall_score(val_targets, val_pred_binary)
        f1 = f1_score(val_targets, val_pred_binary)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'val_loss': best_val_loss
        }
    
    def optimize_hyperparameters(self, book_col, n_folds=5):
        """Optimize hyperparameters for a specific book using cross-validation."""
        book_name = self.book_names[book_col]
        logger.info(f"Optimizing hyperparameters for {book_name}")
        
        # Create balanced dataset
        X_balanced, y_balanced, original_indices = self.create_balanced_dataset(book_col)
        
        # Define hyperparameter grid
        hyperparameter_grid = {
            'hidden_dims': [
                [256, 128, 64],
                [512, 256, 128],
                [256, 128],
                [512, 256, 128, 64],
                [128, 64, 32]
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005, 0.0001, 0.01],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150],
            'patience': [10, 15, 20]
        }
        
        # Generate all combinations
        param_names = list(hyperparameter_grid.keys())
        param_values = list(hyperparameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(combinations)} hyperparameter combinations")
        
        # Cross-validation setup
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        best_score = 0
        best_hyperparams = None
        best_cv_results = None
        
        results = []
        
        # Test each combination
        for i, combination in enumerate(tqdm(combinations, desc=f"Optimizing {book_name}")):
            hyperparams = dict(zip(param_names, combination))
            
            cv_scores = []
            
            # Cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_balanced)):
                X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
                y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                metrics = self.train_model_with_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val, hyperparams
                )
                
                cv_scores.append(metrics['f1_score'])
            
            # Calculate average CV score
            avg_f1 = np.mean(cv_scores)
            std_f1 = np.std(cv_scores)
            
            result = {
                'hyperparams': hyperparams,
                'avg_f1_score': avg_f1,
                'std_f1_score': std_f1,
                'cv_scores': cv_scores
            }
            results.append(result)
            
            # Update best if better
            if avg_f1 > best_score:
                best_score = avg_f1
                best_hyperparams = hyperparams
                best_cv_results = result
        
        logger.info(f"Best F1 score for {book_name}: {best_score:.4f} ± {best_cv_results['std_f1_score']:.4f}")
        logger.info(f"Best hyperparameters: {best_hyperparams}")
        
        return {
            'book_name': book_name,
            'best_hyperparams': best_hyperparams,
            'best_score': best_score,
            'best_cv_results': best_cv_results,
            'all_results': results
        }
    
    def optimize_all_models(self):
        """Optimize hyperparameters for all book models."""
        logger.info("Starting hyperparameter optimization for all models...")
        
        results = {}
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            try:
                result = self.optimize_hyperparameters(book_col)
                results[book_col] = result
            except Exception as e:
                logger.error(f"Error optimizing {book_col}: {e}")
        
        # Save results
        self.save_optimization_results(results)
        
        return results
    
    def save_optimization_results(self, results):
        """Save optimization results to files."""
        # Create results directory
        Path("optimization_results").mkdir(exist_ok=True)
        
        # Save detailed results
        with open('optimization_results/hyperparameter_optimization_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for book_col, result in results.items():
                serializable_results[book_col] = {
                    'book_name': result['book_name'],
                    'best_hyperparams': result['best_hyperparams'],
                    'best_score': float(result['best_score']),
                    'best_cv_results': {
                        'avg_f1_score': float(result['best_cv_results']['avg_f1_score']),
                        'std_f1_score': float(result['best_cv_results']['std_f1_score']),
                        'cv_scores': [float(x) for x in result['best_cv_results']['cv_scores']]
                    }
                }
            json.dump(serializable_results, f, indent=2)
        
        # Create summary report
        self.create_optimization_summary(results)
        
        logger.info("Optimization results saved to optimization_results/")
    
    def create_optimization_summary(self, results):
        """Create a summary report of optimization results."""
        summary = "# Hyperparameter Optimization Results\n\n"
        
        summary += "## Overview\n"
        summary += "Hyperparameter optimization was performed for each individual book model using 5-fold cross-validation.\n\n"
        
        summary += "## Best Hyperparameters by Book\n\n"
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            if book_col in results:
                result = results[book_col]
                book_name = result['book_name']
                best_params = result['best_hyperparams']
                best_score = result['best_score']
                std_score = result['best_cv_results']['std_f1_score']
                
                summary += f"### {book_name}\n\n"
                summary += f"- **Best F1 Score**: {best_score:.4f} ± {std_score:.4f}\n"
                summary += f"- **Hidden Dimensions**: {best_params['hidden_dims']}\n"
                summary += f"- **Dropout Rate**: {best_params['dropout_rate']}\n"
                summary += f"- **Learning Rate**: {best_params['learning_rate']}\n"
                summary += f"- **Batch Size**: {best_params['batch_size']}\n"
                summary += f"- **Epochs**: {best_params['epochs']}\n"
                summary += f"- **Patience**: {best_params['patience']}\n\n"
        
        # Save summary
        with open('optimization_results/optimization_summary.md', 'w') as f:
            f.write(summary)
        
        print(summary)

def main():
    """Main function to run hyperparameter optimization."""
    logger.info("Starting hyperparameter optimization...")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer()
    
    # Optimize all models
    results = optimizer.optimize_all_models()
    
    logger.info("Hyperparameter optimization completed!")
    
    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*60)
    
    for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
        if book_col in results:
            result = results[book_col]
            print(f"\n{result['book_name']}:")
            print(f"  Best F1 Score: {result['best_score']:.4f}")
            print(f"  Best Hyperparameters: {result['best_hyperparams']}")

if __name__ == "__main__":
    main() 