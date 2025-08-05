"""
Hyperparameter Optimization Module for Neural Networks
Supports both multi-label and per-book model optimization using Optuna.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiLabelMLP(nn.Module):
    """Multi-Layer Perceptron for multi-label classification."""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], output_dim=4, dropout_rate=0.3):
        super(MultiLabelMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid for multi-label classification
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

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
    """Hyperparameter optimizer for neural networks using Optuna."""
    
    def __init__(self, 
                 model_type: str = "multi_label",  # "multi_label" or "per_book"
                 dataset_path: str = "data/dataset",
                 embeddings_path: str = "data/embeddings_cache_aligned_f24a423ed8f9dd531230fe64f71f668d.npz",
                 n_trials: int = 50,
                 timeout: int = 3600,
                 study_name: str = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            model_type: "multi_label" or "per_book"
            dataset_path: Path to the dataset
            embeddings_path: Path to the embeddings
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name for the Optuna study
        """
        self.model_type = model_type
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataset = None
        self.embeddings = None
        
        # Load book-to-label mapping from JSON file
        self.book_mapping = self.load_book_mapping()
        
        # Book order in labels list (from the mapping file)
        self.book_order = self.book_mapping['books']
        
        # Book names mapping - using actual column names from the dataset
        # Generate from the loaded mapping to ensure correct order
        self.book_names = {}
        self.book_columns = []
        for book_name in self.book_mapping['books']:
            # Convert book name to column format (replace spaces with underscores, remove apostrophes)
            clean_name = book_name.replace(' ', '_').replace("'", '')
            book_col = f"book_{clean_name}"
            self.book_names[book_col] = book_name
            self.book_columns.append(book_col)
            
        logger.info(f"Initializing {model_type} hyperparameter optimizer")
        logger.info(f"Using device: {self.device}")
        
    def load_book_mapping(self):
        """Load book-to-label mapping from JSON file."""
        mapping_file = Path("data/semantic_augmented/book_to_label_mapping.json")
        if not mapping_file.exists():
            logger.warning(f"Book mapping file not found: {mapping_file}. Using default order.")
            return {
                'books': [
                    'Anna Karenina',
                    'Frankenstein',
                    'The Adventures of Alice in Wonderland',
                    'Wuthering Heights'
                ],
                'book_to_label': {
                    'Anna Karenina': 0,
                    'Frankenstein': 1,
                    'The Adventures of Alice in Wonderland': 2,
                    'Wuthering Heights': 3
                },
                'label_to_book': {
                    '0': 'Anna Karenina',
                    '1': 'Frankenstein',
                    '2': 'The Adventures of Alice in Wonderland',
                    '3': 'Wuthering Heights'
                },
                'num_classes': 4
            }
        
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            logger.info(f"Loaded book mapping from: {mapping_file}")
            logger.info(f"Book order: {mapping['books']}")
            return mapping
        except Exception as e:
            logger.error(f"Failed to load book mapping: {e}. Using default order.")
            return {
                'books': [
                    'Anna Karenina',
                    'Frankenstein',
                    'The Adventures of Alice in Wonderland',
                    'Wuthering Heights'
                ],
                'book_to_label': {
                    'Anna Karenina': 0,
                    'Frankenstein': 1,
                    'The Adventures of Alice in Wonderland': 2,
                    'Wuthering Heights': 3
                },
                'label_to_book': {
                    '0': 'Anna Karenina',
                    '1': 'Frankenstein',
                    '2': 'The Adventures of Alice in Wonderland',
                    '3': 'Wuthering Heights'
                },
                'num_classes': 4
            }
        
    def load_data(self):
        """Load dataset and embeddings."""
        logger.info("Loading dataset and embeddings...")
        
        # Load dataset splits
        self.dataset = load_from_disk(self.dataset_path)
        logger.info(f"Loaded dataset splits:")
        logger.info(f"  Train: {len(self.dataset['train'])} samples")
        logger.info(f"  Validation: {len(self.dataset['validation'])} samples")
        logger.info(f"  Test: {len(self.dataset['test'])} samples")
        
        # Load embeddings
        embeddings_data = np.load(self.embeddings_path)
        self.embeddings = embeddings_data['embeddings'].astype(np.float32)
        logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
        
        return True
    
    def get_multi_label_data(self):
        """Get data for multi-label model."""
        # Extract multi-label data from all splits
        train_labels = np.array(self.dataset['train']['labels'], dtype=np.float32)
        val_labels = np.array(self.dataset['validation']['labels'], dtype=np.float32)
        test_labels = np.array(self.dataset['test']['labels'], dtype=np.float32)
        
        # Get embeddings for each split
        train_embeddings = self.embeddings[:len(self.dataset['train'])]
        val_embeddings = self.embeddings[len(self.dataset['train']):len(self.dataset['train']) + len(self.dataset['validation'])]
        test_embeddings = self.embeddings[len(self.dataset['train']) + len(self.dataset['validation']):]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_embeddings)
        X_val_scaled = scaler.transform(val_embeddings)
        X_test_scaled = scaler.transform(test_embeddings)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        X_val = torch.FloatTensor(X_val_scaled).to(self.device)
        X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train = torch.FloatTensor(train_labels).to(self.device)
        y_val = torch.FloatTensor(val_labels).to(self.device)
        y_test = torch.FloatTensor(test_labels).to(self.device)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    
    def get_per_book_data(self, book_col: str):
        """Get data for a specific book."""
        book_name = self.book_names[book_col]
        book_idx = self.book_order.index(book_name)
        
        # Extract labels for this book from all splits
        train_labels = [labels[book_idx] for labels in self.dataset['train']['labels']]
        val_labels = [labels[book_idx] for labels in self.dataset['validation']['labels']]
        test_labels = [labels[book_idx] for labels in self.dataset['test']['labels']]
        
        # Get embeddings for each split
        train_embeddings = self.embeddings[:len(self.dataset['train'])]
        val_embeddings = self.embeddings[len(self.dataset['train']):len(self.dataset['train']) + len(self.dataset['validation'])]
        test_embeddings = self.embeddings[len(self.dataset['train']) + len(self.dataset['validation']):]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_embeddings)
        X_val_scaled = scaler.transform(val_embeddings)
        X_test_scaled = scaler.transform(test_embeddings)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        X_val = torch.FloatTensor(X_val_scaled).to(self.device)
        X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train = torch.FloatTensor(train_labels).to(self.device)
        y_val = torch.FloatTensor(val_labels).to(self.device)
        y_test = torch.FloatTensor(test_labels).to(self.device)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    
    def suggest_multi_label_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for multi-label model."""
        # Architecture hyperparameters
        n_layers = trial.suggest_int("n_layers", 5, 10)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 64, 512)
            hidden_dims.append(hidden_dim)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Training duration
        epochs = trial.suggest_int("epochs", 50, 200)
        patience = trial.suggest_int("patience", 10, 30)
        
        return {
            'hidden_dims': hidden_dims,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'patience': patience
        }
    
    def suggest_per_book_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for per-book model."""
        # Architecture hyperparameters
        n_layers = trial.suggest_int("n_layers", 8, 12)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 64, 512)
            hidden_dims.append(hidden_dim)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Training duration
        epochs = trial.suggest_int("epochs", 50, 200)
        patience = trial.suggest_int("patience", 10, 30)
        
        return {
            'hidden_dims': hidden_dims,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'patience': patience
        }
    
    def train_multi_label_model(self, trial: optuna.Trial) -> float:
        """Train and evaluate multi-label model for a trial."""
        # Get hyperparameters
        params = self.suggest_multi_label_hyperparameters(trial)
        
        # Get data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.get_multi_label_data()
        
        # Create model
        model = MultiLabelMLP(
            input_dim=384,
            hidden_dims=params['hidden_dims'],
            output_dim=4,
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_outputs.append(outputs.cpu())
                    val_targets.append(batch_y.cpu())
                
                # Calculate validation F1
                val_outputs = torch.cat(val_outputs).numpy()
                val_targets = torch.cat(val_targets).numpy()
                val_pred = (val_outputs > 0.5).astype(int)
                val_f1 = f1_score(val_targets, val_pred, average='weighted', zero_division=0)
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= params['patience']:
                break
        
        return best_val_f1
    
    def train_per_book_model(self, trial: optuna.Trial, book_col: str) -> float:
        """Train and evaluate per-book model for a trial."""
        # Get hyperparameters
        params = self.suggest_per_book_hyperparameters(trial)
        
        # Get data for this book
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.get_per_book_data(book_col)
        
        # Create model
        model = BinaryBookClassifier(
            input_dim=384,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_outputs.append(outputs.cpu())
                    val_targets.append(batch_y.cpu())
                
                # Calculate validation F1
                val_outputs = torch.cat(val_outputs).numpy()
                val_targets = torch.cat(val_targets).numpy()
                val_pred = (val_outputs > 0.5).astype(int)
                val_f1 = f1_score(val_targets, val_pred, zero_division=0)
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= params['patience']:
                break
        
        return best_val_f1
    
    def optimize_multi_label(self) -> optuna.Study:
        """Optimize hyperparameters for multi-label model."""
        logger.info("Starting multi-label hyperparameter optimization...")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=None  # In-memory storage
        )
        
        # Optimize
        study.optimize(
            self.train_multi_label_model,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        logger.info(f"Multi-label optimization completed!")
        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        return study
    
    def optimize_per_book(self, book_col: str) -> optuna.Study:
        """Optimize hyperparameters for a specific book."""
        book_name = self.book_names[book_col]
        logger.info(f"Starting hyperparameter optimization for {book_name}...")
        
        # Create study
        study_name = f"{self.study_name}_{book_col}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=None  # In-memory storage
        )
        
        # Create objective function for this book
        def objective(trial):
            return self.train_per_book_model(trial, book_col)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        logger.info(f"{book_name} optimization completed!")
        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        return study
    
    def optimize_specific_book(self, book_name: str) -> optuna.Study:
        """Optimize hyperparameters for a specific book by name."""
        # Find the book column for the given book name
        book_col = None
        for col, name in self.book_names.items():
            if name == book_name:
                book_col = col
                break
        
        if book_col is None:
            raise ValueError(f"Book '{book_name}' not found. Available books: {list(self.book_names.values())}")
        
        return self.optimize_per_book(book_col)
    
    def optimize_all_books(self) -> Dict[str, optuna.Study]:
        """Optimize hyperparameters for all books."""
        logger.info("Starting hyperparameter optimization for all books...")
        
        studies = {}
        for book_col in self.book_columns:
            study = self.optimize_per_book(book_col)
            studies[book_col] = study
        
        return studies
    
    def create_optimization_report(self, study: optuna.Study, model_type: str, book_name: str = None) -> str:
        """Create a detailed optimization report."""
        report = []
        report.append(f"# Hyperparameter Optimization Report")
        report.append(f"")
        report.append(f"**Model Type**: {model_type}")
        if book_name:
            report.append(f"**Book**: {book_name}")
        report.append(f"**Study Name**: {study.study_name}")
        report.append(f"**Best Trial Value**: {study.best_trial.value:.4f}")
        report.append(f"**Number of Trials**: {len(study.trials)}")
        report.append(f"")
        
        # Best parameters
        report.append("## Best Parameters")
        report.append("```json")
        report.append(json.dumps(study.best_trial.params, indent=2))
        report.append("```")
        report.append("")
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            report.append("## Parameter Importance")
            for param, imp in importance.items():
                report.append(f"- **{param}**: {imp:.4f}")
            report.append("")
        except:
            report.append("## Parameter Importance")
            report.append("Could not calculate parameter importance (insufficient trials)")
            report.append("")
        
        # Trial history
        report.append("## Trial History")
        report.append("| Trial | Value | Status |")
        report.append("|-------|-------|--------|")
        for trial in study.trials:
            status = "COMPLETE" if trial.state == optuna.trial.TrialState.COMPLETE else "FAILED"
            report.append(f"| {trial.number} | {trial.value:.4f} | {status} |")
        report.append("")
        
        return "\n".join(report)
    
    def save_optimization_results(self, study: optuna.Study, model_type: str, book_name: str = None):
        """Save optimization results to files."""
        # Create results directory
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save study
        study_name = study.study_name
        study_path = results_dir / f"{study_name}.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        # Save report
        report = self.create_optimization_report(study, model_type, book_name)
        report_path = results_dir / f"{study_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save best parameters
        params_path = results_dir / f"{study_name}_best_params.json"
        with open(params_path, 'w') as f:
            json.dump(study.best_trial.params, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_dir}")
        logger.info(f"Study: {study_path}")
        logger.info(f"Report: {report_path}")
        logger.info(f"Best params: {params_path}")
    
    def plot_optimization_history(self, study: optuna.Study, save_path: str = None):
        """Plot optimization history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Optimization history
        values = [trial.value for trial in study.trials if trial.value is not None]
        ax1.plot(values)
        ax1.set_title('Optimization History')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Objective Value')
        ax1.grid(True)
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            ax2.barh(params, importances)
            ax2.set_title('Parameter Importance')
            ax2.set_xlabel('Importance')
            ax2.grid(True)
        except:
            ax2.text(0.5, 0.5, 'Insufficient trials\nfor importance calculation', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization plot saved to {save_path}")
        
        plt.show()

def optimize_multi_label_model(n_trials: int = 50, timeout: int = 3600):
    """Optimize hyperparameters for multi-label model."""
    optimizer = HyperparameterOptimizer(
        model_type="multi_label",
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Load data
    optimizer.load_data()
    
    # Run optimization
    study = optimizer.optimize_multi_label()
    
    # Save results
    optimizer.save_optimization_results(study, "multi_label")
    
    # Create plot
    optimizer.plot_optimization_history(study, "optimization_results/multi_label_optimization.png")
    
    return study

def optimize_per_book_models(n_trials: int = 50, timeout: int = 3600):
    """Optimize hyperparameters for all per-book models."""
    optimizer = HyperparameterOptimizer(
        model_type="per_book",
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Load data
    optimizer.load_data()
    
    # Run optimization for all books
    studies = optimizer.optimize_all_books()
    
    # Save results for each book
    for book_col, study in studies.items():
        book_name = optimizer.book_names[book_col]
        optimizer.save_optimization_results(study, "per_book", book_name)
        optimizer.plot_optimization_history(
            study, 
            f"optimization_results/{book_col}_optimization.png"
        )
    
    return studies

def optimize_specific_book_model(book_name: str, n_trials: int = 50, timeout: int = 3600):
    """Optimize hyperparameters for a specific book model."""
    optimizer = HyperparameterOptimizer(
        model_type="per_book",
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Load data
    optimizer.load_data()
    
    # Run optimization for specific book
    study = optimizer.optimize_specific_book(book_name)
    
    # Save results
    optimizer.save_optimization_results(study, "per_book", book_name)
    optimizer.plot_optimization_history(
        study, 
        f"optimization_results/{book_name.replace(' ', '_').lower()}_optimization.png"
    )
    
    return study

if __name__ == "__main__":
    # Example usage
    import argparse
    
    # Create optimizer to get available book names
    optimizer = HyperparameterOptimizer()
    available_books = optimizer.book_mapping['books']
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--model_type", choices=["multi_label", "per_book", "specific_book"], 
                       default="multi_label", help="Model type to optimize")
    parser.add_argument("--book_name", type=str, 
                       choices=available_books,
                       help="Specific book name for optimization (required for specific_book mode)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    if args.model_type == "multi_label":
        study = optimize_multi_label_model(args.n_trials, args.timeout)
        print(f"Best F1 Score: {study.best_trial.value:.4f}")
    elif args.model_type == "per_book":
        studies = optimize_per_book_models(args.n_trials, args.timeout)
        for book_col, study in studies.items():
            book_name = optimizer.book_names[book_col]
            print(f"{book_name}: Best F1 Score: {study.best_trial.value:.4f}")
    elif args.model_type == "specific_book":
        if not args.book_name:
            print("Error: --book_name is required for specific_book mode")
            print(f"Available books: {', '.join(available_books)}")
            exit(1)
        
        study = optimize_specific_book_model(args.book_name, args.n_trials, args.timeout)
        print(f"{args.book_name}: Best F1 Score: {study.best_trial.value:.4f}") 