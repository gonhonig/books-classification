"""
Unified Hyperparameter Optimization for Individual Book Models
Uses the same unified dataset approach to ensure consistent evaluation across all models.
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

class UnifiedHyperparameterOptimizer:
    """Hyperparameter optimizer using unified dataset approach."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Book names mapping
        self.book_names = {
            'book_1': 'Anna Karenina',
            'book_2': 'Wuthering Heights', 
            'book_3': 'Frankenstein',
            'book_4': 'The Adventures of Alice in Wonderland'
        }
        
        # Load data once
        self.load_data()
        self.create_unified_dataset()
        
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load the semantic augmented dataset and embeddings."""
        logger.info("Loading semantic augmented dataset and embeddings...")
        
        # Load the semantic augmented dataset
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Load embeddings from cache
        embeddings_data = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')
        self.embeddings = embeddings_data['embeddings'].astype(np.float32)
        logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
        
        # Verify that embeddings match dataset size
        if len(self.embeddings) != len(df):
            raise ValueError(f"Embeddings count ({len(self.embeddings)}) doesn't match dataset size ({len(df)})")
        
        # Extract book labels - map the column names to our expected format
        self.book_labels = {}
        column_mapping = {
            'book_Anna_Karenina': 'book_1',
            'book_Wuthering_Heights': 'book_2', 
            'book_Frankenstein': 'book_3',
            'book_The_Adventures_of_Alice_in_Wonderland': 'book_4'
        }
        
        for col_name, book_col in column_mapping.items():
            if col_name in df.columns:
                self.book_labels[book_col] = df[col_name].values.astype(np.float32)
                logger.info(f"Loaded {col_name} -> {book_col}: {np.sum(self.book_labels[book_col] == 1)} positive samples")
            else:
                logger.warning(f"Column {col_name} not found in dataset")
                self.book_labels[book_col] = np.zeros(len(df), dtype=np.float32)
        
        logger.info("Data loaded successfully")
        
    def create_unified_dataset(self, target_samples_per_class=5000):
        """Create a unified balanced dataset for all books, prioritizing multi-label sentences."""
        logger.info("Creating unified balanced dataset for optimization...")
        
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
        
        logger.info(f"Found {len(multi_label_indices)} multi-label sentences")
        
        # Include ALL multi-label sentences
        selected_indices = multi_label_indices.copy()
        logger.info(f"Included {len(selected_indices)} multi-label sentences")
        
        # For each book, add single-label positive samples to reach target
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            # Get single-label positive samples for this book
            positive_indices = np.where(self.book_labels[book_col] == 1)[0]
            single_label_positive = [i for i in positive_indices if i not in multi_label_indices]
            
            # Calculate how many more positive samples we need
            current_positive_count = sum(1 for i in selected_indices if self.book_labels[book_col][i] == 1)
            additional_needed = target_samples_per_class - current_positive_count
            
            if additional_needed > 0 and len(single_label_positive) > 0:
                # Sample additional single-label positive samples
                n_to_sample = min(additional_needed, len(single_label_positive))
                np.random.seed(42)  # For reproducibility
                selected_single_label = np.random.choice(single_label_positive, n_to_sample, replace=False)
                selected_indices.extend(selected_single_label)
                logger.info(f"Added {n_to_sample} single-label positive samples for {self.book_names[book_col]}")
        
        # Add negative samples to balance the dataset
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            positive_count = sum(1 for i in selected_indices if self.book_labels[book_col][i] == 1)
            negative_count = sum(1 for i in selected_indices if self.book_labels[book_col][i] == 0)
            
            if negative_count < positive_count:
                # Need more negative samples
                additional_negative_needed = positive_count - negative_count
                negative_indices = np.where(self.book_labels[book_col] == 0)[0]
                available_negative = [i for i in negative_indices if i not in selected_indices]
                
                if len(available_negative) > 0:
                    n_to_sample = min(additional_negative_needed, len(available_negative))
                    np.random.seed(42)
                    selected_negative = np.random.choice(available_negative, n_to_sample, replace=False)
                    selected_indices.extend(selected_negative)
                    logger.info(f"Added {n_to_sample} negative samples for {self.book_names[book_col]}")
        
        # Remove duplicates and shuffle
        selected_indices = list(set(selected_indices))
        np.random.seed(42)
        np.random.shuffle(selected_indices)
        
        # Create the unified dataset
        self.X_unified = self.embeddings[selected_indices]
        self.y_unified = {}
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            self.y_unified[book_col] = self.book_labels[book_col][selected_indices]
        
        # Store selected indices for later use
        self.selected_indices = selected_indices
        
        # Log dataset statistics
        logger.info(f"Created unified dataset with {len(self.X_unified)} samples")
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            positive_count = np.sum(self.y_unified[book_col] == 1)
            negative_count = np.sum(self.y_unified[book_col] == 0)
            multi_label_count = sum(1 for i in selected_indices if i in multi_label_indices)
            logger.info(f"{self.book_names[book_col]}: {positive_count} positive, {negative_count} negative, {multi_label_count} multi-label")
    
    def train_model_with_hyperparameters(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train a model with given hyperparameters and return validation metrics."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Create model
        model = BinaryBookClassifier(
            input_dim=384,
            hidden_dims=hyperparams['hidden_dims'],
            dropout_rate=hyperparams['dropout_rate']
        ).to(self.device)
        
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
            
            # Learning rate scheduling
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
        val_pred_binary = (np.array(val_predictions) > 0.5).astype(int)
        val_targets = np.array(val_targets)
        
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
        """Optimize hyperparameters for a specific book using K-Fold cross-validation."""
        book_name = self.book_names[book_col]
        logger.info(f"Optimizing hyperparameters for {book_name}")
        
        # Define hyperparameter grid
        hyperparameter_grid = {
            'hidden_dims': [
                [256, 128, 64],
                [512, 256, 128],
                [256, 128],
                [512, 256, 128, 64],
                [128, 64, 32]
            ],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100],
            'patience': [10, 15, 20]
        }
        
        # Generate all combinations
        param_names = list(hyperparameter_grid.keys())
        param_values = list(hyperparameter_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(all_combinations)} hyperparameter combinations")
        
        # Get labels for this book
        y_book = self.y_unified[book_col]
        
        # K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        best_f1_score = 0.0
        best_hyperparams = None
        best_metrics = None
        
        # Progress bar for combinations
        for combination in tqdm(all_combinations, desc=f"Optimizing {book_name}"):
            hyperparams = dict(zip(param_names, combination))
            
            fold_scores = []
            
            # K-Fold evaluation
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_unified)):
                X_train, X_val = self.X_unified[train_idx], self.X_unified[val_idx]
                y_train, y_val = y_book[train_idx], y_book[val_idx]
                
                try:
                    metrics = self.train_model_with_hyperparameters(X_train, y_train, X_val, y_val, hyperparams)
                    fold_scores.append(metrics['f1_score'])
                except Exception as e:
                    logger.warning(f"Error in fold {fold} with hyperparams {hyperparams}: {e}")
                    fold_scores.append(0.0)
            
            # Calculate average F1 score across folds
            avg_f1 = np.mean(fold_scores)
            
            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_hyperparams = hyperparams
                best_metrics = {
                    'f1_score': avg_f1,
                    'fold_scores': fold_scores,
                    'std_f1': np.std(fold_scores)
                }
        
        logger.info(f"Best F1 score for {book_name}: {best_f1_score:.4f}")
        logger.info(f"Best hyperparameters: {best_hyperparams}")
        
        return {
            'book_name': book_name,
            'best_f1_score': best_f1_score,
            'best_hyperparams': best_hyperparams,
            'best_metrics': best_metrics
        }
    
    def optimize_all_models(self):
        """Optimize hyperparameters for all book models."""
        logger.info("Starting hyperparameter optimization for all models...")
        
        # Create results directory
        Path("optimization_results").mkdir(exist_ok=True)
        
        results = {}
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            try:
                result = self.optimize_hyperparameters(book_col)
                results[book_col] = result
            except Exception as e:
                logger.error(f"Error optimizing hyperparameters for {book_col}: {e}")
        
        # Save detailed results
        self.save_optimization_results(results)
        
        # Create summary report
        self.create_optimization_summary(results)
        
        return results
    
    def save_optimization_results(self, results):
        """Save detailed optimization results."""
        results_path = "optimization_results/unified_hyperparameter_optimization_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for book_col, result in results.items():
            serializable_results[book_col] = {
                'book_name': result['book_name'],
                'best_f1_score': float(result['best_f1_score']),
                'best_hyperparams': result['best_hyperparams'],
                'best_metrics': {
                    'f1_score': float(result['best_metrics']['f1_score']),
                    'fold_scores': [float(score) for score in result['best_metrics']['fold_scores']],
                    'std_f1': float(result['best_metrics']['std_f1'])
                }
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Detailed optimization results saved to {results_path}")
    
    def create_optimization_summary(self, results):
        """Create a summary report of optimization results."""
        summary_path = "optimization_results/unified_hyperparameter_optimization_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write("# Unified Hyperparameter Optimization Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Book | Best F1 Score | Std F1 | Best Hyperparameters |\n")
            f.write("|------|---------------|--------|---------------------|\n")
            
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                if book_col in results:
                    result = results[book_col]
                    book_name = result['book_name']
                    f1_score = result['best_f1_score']
                    std_f1 = result['best_metrics']['std_f1']
                    hyperparams = result['best_hyperparams']
                    
                    # Format hyperparameters for display
                    hyperparams_str = f"hidden_dims: {hyperparams['hidden_dims']}, "
                    hyperparams_str += f"dropout: {hyperparams['dropout_rate']}, "
                    hyperparams_str += f"lr: {hyperparams['learning_rate']}, "
                    hyperparams_str += f"batch_size: {hyperparams['batch_size']}, "
                    hyperparams_str += f"epochs: {hyperparams['epochs']}, "
                    hyperparams_str += f"patience: {hyperparams['patience']}"
                    
                    f.write(f"| {book_name} | {f1_score:.4f} | {std_f1:.4f} | {hyperparams_str} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                if book_col in results:
                    result = results[book_col]
                    f.write(f"### {result['book_name']}\n\n")
                    f.write(f"- **Best F1 Score**: {result['best_f1_score']:.4f}\n")
                    f.write(f"- **Standard Deviation**: {result['best_metrics']['std_f1']:.4f}\n")
                    f.write(f"- **Fold Scores**: {[f'{score:.4f}' for score in result['best_metrics']['fold_scores']]}\n\n")
                    
                    f.write("**Best Hyperparameters**:\n")
                    for param, value in result['best_hyperparams'].items():
                        f.write(f"- {param}: {value}\n")
                    f.write("\n")
        
        logger.info(f"Optimization summary saved to {summary_path}")

def main():
    """Main optimization function."""
    logger.info("Starting unified hyperparameter optimization...")
    
    # Initialize optimizer
    optimizer = UnifiedHyperparameterOptimizer()
    
    # Optimize all models
    results = optimizer.optimize_all_models()
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED")
    logger.info("=" * 60)
    
    # Print summary
    for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
        if book_col in results:
            result = results[book_col]
            logger.info(f"{result['book_name']}: F1 = {result['best_f1_score']:.4f}")
    
    logger.info("Optimization completed!")

if __name__ == "__main__":
    main() 