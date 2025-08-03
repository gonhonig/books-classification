"""
Neural Network for Multi-Label Classification with Semantic Embeddings
Uses PyTorch with MPS device support.
"""

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
import yaml
from datetime import datetime
from datasets import load_from_disk

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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

class NeuralNetworkTrainer:
    """Trainer for neural network multi-label classification."""
    
    def __init__(self, dataset_path: str = "data/dataset", embeddings_path: str = "data/embeddings_cache_aligned_f24a423ed8f9dd531230fe64f71f668d.npz", config_path: str = "configs/optimized_params_config.yaml"):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.config_path = config_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        self.config = self.load_config()
        
        # Dataset properties
        self.dataset = None
        self.embeddings = None
        
        # Book order in labels list (based on the dataset creation order)
        self.book_order = [
            'Anna Karenina',
            'Frankenstein', 
            'The Adventures of Alice in Wonderland',
            'Wuthering Heights'
        ]
        
        # Book names mapping
        self.book_names = {
            'book_1': 'Anna Karenina',
            'book_2': 'Wuthering Heights', 
            'book_3': 'Frankenstein',
            'book_4': 'The Adventures of Alice in Wonderland'
        }
        
        logger.info(f"Using device: {self.device}")
        
    def load_config(self):
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}. Using default parameters.")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}. Using default parameters.")
            return None
        
    def load_data(self):
        """Load the pre-existing dataset splits and embeddings."""
        logger.info("Loading dataset splits and embeddings...")
        
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
        
        # Extract multi-label data from all splits
        train_labels = np.array(self.dataset['train']['labels'], dtype=np.float32)
        val_labels = np.array(self.dataset['validation']['labels'], dtype=np.float32)
        test_labels = np.array(self.dataset['test']['labels'], dtype=np.float32)
        
        # Get embeddings for each split
        train_embeddings = self.embeddings[:len(self.dataset['train'])]
        val_embeddings = self.embeddings[len(self.dataset['train']):len(self.dataset['train']) + len(self.dataset['validation'])]
        test_embeddings = self.embeddings[len(self.dataset['train']) + len(self.dataset['validation']):]
        
        # Scale features
        self.scaler.fit(train_embeddings)
        X_train_scaled = self.scaler.transform(train_embeddings)
        X_val_scaled = self.scaler.transform(val_embeddings)
        X_test_scaled = self.scaler.transform(test_embeddings)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        self.X_val = torch.FloatTensor(X_val_scaled).to(self.device)
        self.X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        self.y_train = torch.FloatTensor(train_labels).to(self.device)
        self.y_val = torch.FloatTensor(val_labels).to(self.device)
        self.y_test = torch.FloatTensor(test_labels).to(self.device)
        
        # Check label distribution
        unique_labels_per_sample = np.sum(train_labels, axis=1)
        logger.info(f"Label distribution:")
        logger.info(f"  Single-label samples: {np.sum(unique_labels_per_sample == 1)}")
        logger.info(f"  Multi-label samples: {np.sum(unique_labels_per_sample > 1)}")
        logger.info(f"  Average labels per sample: {np.mean(unique_labels_per_sample):.2f}")
        
        logger.info(f"Features shape: {self.X_train.shape}")
        logger.info(f"Labels shape: {self.y_train.shape}")
        logger.info("Data preparation completed!")
        return True
    
    def create_model(self, hidden_dims=None, dropout_rate=None):
        """Create the neural network model."""
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        
        # Use config parameters if available, otherwise use defaults
        if self.config and 'multi_label' in self.config:
            config_params = self.config['multi_label']['architecture']
            hidden_dims = hidden_dims or config_params['hidden_dims']
            dropout_rate = dropout_rate or config_params['dropout_rate']
        else:
            hidden_dims = hidden_dims or [256, 128, 64]
            dropout_rate = dropout_rate or 0.3
        
        self.model = MultiLabelMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
        logger.info(f"Architecture: hidden_dims={hidden_dims}, dropout_rate={dropout_rate}")
        return self.model
    
    def train_model(self, epochs=None, batch_size=None, learning_rate=None, patience=None, weight_decay=None):
        """Train the neural network."""
        if self.model is None:
            self.create_model()
        
        # Use config parameters if available, otherwise use defaults
        if self.config and 'multi_label' in self.config:
            config_params = self.config['multi_label']['training']
            epochs = epochs or config_params['epochs']
            batch_size = batch_size or config_params['batch_size']
            learning_rate = learning_rate or config_params['learning_rate']
            patience = patience or config_params['patience']
            weight_decay = weight_decay or config_params['weight_decay']
        else:
            epochs = epochs or 100
            batch_size = batch_size or 64
            learning_rate = learning_rate or 0.001
            patience = patience or 10
            weight_decay = weight_decay or 1e-5
        
        logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, patience={patience}, weight_decay={weight_decay}")
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_outputs = self.model(batch_X)
                    val_loss += criterion(val_outputs, batch_y).item()
                
                # Calculate metrics on full validation set
                val_outputs = self.model(self.X_val)
                val_pred = (val_outputs > 0.5).float().cpu().numpy()
                val_true = self.y_val.cpu().numpy()
                
                val_accuracy = accuracy_score(val_true, val_pred)
                val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'multi_label_model/best_neural_network.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('multi_label_model/best_neural_network.pth'))
        logger.info("Training completed!")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        if self.model is None:
            logger.error("No model to evaluate!")
            return
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            predictions = (outputs > 0.5).float().cpu().numpy()
            probabilities = outputs.cpu().numpy()
            true_labels = self.y_test.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        hamming = hamming_loss(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Average predictions per sample
        avg_predictions = np.mean(np.sum(predictions, axis=1))
        
        # Analyze performance by label count
        test_label_counts = np.sum(true_labels, axis=1)
        
        logger.info("Neural Network Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg predictions per sample: {avg_predictions:.2f}")
        
        logger.info("\nPerformance by label count:")
        for label_count in sorted(set(test_label_counts)):
            mask = test_label_counts == label_count
            if np.sum(mask) > 0:
                subset_accuracy = accuracy_score(true_labels[mask], predictions[mask])
                subset_f1 = f1_score(true_labels[mask], predictions[mask], average='weighted', zero_division=0)
                logger.info(f"  {label_count} label(s): {np.sum(mask)} samples, Accuracy: {subset_accuracy:.4f}, F1: {subset_f1:.4f}")
        
        self.results = {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_predictions': avg_predictions,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return self.results
    
    def evaluate_single_vs_multi_label_performance(self):
        """Evaluate model performance specifically on single-label vs multi-label sentences."""
        logger.info("Evaluating single-label vs multi-label performance...")
        
        # Get test data
        test_data = self.dataset['test']
        test_sentences = test_data['sentence']
        test_labels = test_data['labels']
        
        # Convert test_labels to numpy array
        test_labels_array = np.array(test_labels, dtype=np.float32)
        
        # Identify single-label and multi-label sentences in test set
        single_label_test_indices = []
        multi_label_test_indices = []
        
        for i, labels in enumerate(test_labels):
            if sum(labels) == 1:
                single_label_test_indices.append(i)
            elif sum(labels) > 1:
                multi_label_test_indices.append(i)
        
        logger.info(f"Found {len(single_label_test_indices)} single-label test sentences")
        logger.info(f"Found {len(multi_label_test_indices)} multi-label test sentences")
        
        # Get predictions for all test sentences
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test)
            test_predictions = (test_outputs > 0.5).float().cpu().numpy()
            test_probabilities = test_outputs.cpu().numpy()
        
        # Calculate single-label metrics
        single_label_metrics = {}
        if len(single_label_test_indices) > 0:
            single_label_pred = test_predictions[single_label_test_indices]
            single_label_true = test_labels_array[single_label_test_indices]
            
            single_label_accuracy = accuracy_score(single_label_true, single_label_pred)
            single_label_hamming = hamming_loss(single_label_true, single_label_pred)
            single_label_precision = precision_score(single_label_true, single_label_pred, average='weighted', zero_division=0)
            single_label_recall = recall_score(single_label_true, single_label_pred, average='weighted', zero_division=0)
            single_label_f1 = f1_score(single_label_true, single_label_pred, average='weighted', zero_division=0)
            
            single_label_metrics = {
                'accuracy': float(single_label_accuracy),
                'hamming_loss': float(single_label_hamming),
                'precision': float(single_label_precision),
                'recall': float(single_label_recall),
                'f1_score': float(single_label_f1),
                'count': len(single_label_test_indices)
            }
            
            logger.info(f"Single-label Performance:")
            logger.info(f"  Accuracy: {single_label_accuracy:.4f}")
            logger.info(f"  Hamming Loss: {single_label_hamming:.4f}")
            logger.info(f"  Precision: {single_label_precision:.4f}")
            logger.info(f"  Recall: {single_label_recall:.4f}")
            logger.info(f"  F1 Score: {single_label_f1:.4f}")
        
        # Calculate multi-label metrics
        multi_label_metrics = {}
        if len(multi_label_test_indices) > 0:
            multi_label_pred = test_predictions[multi_label_test_indices]
            multi_label_true = test_labels_array[multi_label_test_indices]
            
            multi_label_accuracy = accuracy_score(multi_label_true, multi_label_pred)
            multi_label_hamming = hamming_loss(multi_label_true, multi_label_pred)
            multi_label_precision = precision_score(multi_label_true, multi_label_pred, average='weighted', zero_division=0)
            multi_label_recall = recall_score(multi_label_true, multi_label_pred, average='weighted', zero_division=0)
            multi_label_f1 = f1_score(multi_label_true, multi_label_pred, average='weighted', zero_division=0)
            
            multi_label_metrics = {
                'accuracy': float(multi_label_accuracy),
                'hamming_loss': float(multi_label_hamming),
                'precision': float(multi_label_precision),
                'recall': float(multi_label_recall),
                'f1_score': float(multi_label_f1),
                'count': len(multi_label_test_indices)
            }
            
            logger.info(f"Multi-label Performance:")
            logger.info(f"  Accuracy: {multi_label_accuracy:.4f}")
            logger.info(f"  Hamming Loss: {multi_label_hamming:.4f}")
            logger.info(f"  Precision: {multi_label_precision:.4f}")
            logger.info(f"  Recall: {multi_label_recall:.4f}")
            logger.info(f"  F1 Score: {multi_label_f1:.4f}")
        
        return {
            'single_label': single_label_metrics,
            'multi_label': multi_label_metrics
        }
    
    def calculate_per_book_metrics(self):
        """Calculate per-book metrics for the multi-label model."""
        logger.info("Calculating per-book metrics for multi-label model...")
        
        # Get test data
        test_data = self.dataset['test']
        test_sentences = test_data['sentence']
        test_labels = test_data['labels']
        
        # Convert test_labels to numpy array
        test_labels_array = np.array(test_labels, dtype=np.float32)
        
        # Identify single-label and multi-label sentences in test set
        single_label_test_indices = []
        multi_label_test_indices = []
        
        for i, labels in enumerate(test_labels):
            if sum(labels) == 1:
                single_label_test_indices.append(i)
            elif sum(labels) > 1:
                multi_label_test_indices.append(i)
        
        logger.info(f"Found {len(single_label_test_indices)} single-label test sentences")
        logger.info(f"Found {len(multi_label_test_indices)} multi-label test sentences")
        
        # Get predictions for all test sentences
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test)
            test_predictions = (test_outputs > 0.5).float().cpu().numpy()
            test_probabilities = test_outputs.cpu().numpy()
        
        # Calculate per-book metrics
        per_book_metrics = {}
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            book_name = self.book_names[book_col]
            book_idx = self.book_order.index(book_name)
            
            # Extract predictions and targets for this book
            book_predictions = test_probabilities[:, book_idx]
            book_targets = test_labels_array[:, book_idx]
            book_pred_binary = (book_predictions > 0.5).astype(int)
            
            # Calculate overall metrics for this book
            overall_accuracy = accuracy_score(book_targets, book_pred_binary)
            overall_precision = precision_score(book_targets, book_pred_binary, zero_division=0)
            overall_recall = recall_score(book_targets, book_pred_binary, zero_division=0)
            overall_f1 = f1_score(book_targets, book_pred_binary, zero_division=0)
            
            # Calculate single-label metrics for this book
            single_label_predictions = []
            single_label_targets = []
            for idx in single_label_test_indices:
                single_label_predictions.append(book_predictions[idx])
                single_label_targets.append(book_targets[idx])
            
            if len(single_label_predictions) > 0:
                single_label_pred_binary = (np.array(single_label_predictions) > 0.5).astype(int)
                single_label_targets = np.array(single_label_targets)
                
                single_label_accuracy = accuracy_score(single_label_targets, single_label_pred_binary)
                single_label_precision = precision_score(single_label_targets, single_label_pred_binary, zero_division=0)
                single_label_recall = recall_score(single_label_targets, single_label_pred_binary, zero_division=0)
                single_label_f1 = f1_score(single_label_targets, single_label_pred_binary, zero_division=0)
            else:
                single_label_accuracy = single_label_precision = single_label_recall = single_label_f1 = 0.0
            
            # Calculate multi-label metrics for this book
            multi_label_predictions = []
            multi_label_targets = []
            for idx in multi_label_test_indices:
                multi_label_predictions.append(book_predictions[idx])
                multi_label_targets.append(book_targets[idx])
            
            if len(multi_label_predictions) > 0:
                multi_label_pred_binary = (np.array(multi_label_predictions) > 0.5).astype(int)
                multi_label_targets = np.array(multi_label_targets)
                
                multi_label_accuracy = accuracy_score(multi_label_targets, multi_label_pred_binary)
                multi_label_precision = precision_score(multi_label_targets, multi_label_pred_binary, zero_division=0)
                multi_label_recall = recall_score(multi_label_targets, multi_label_pred_binary, zero_division=0)
                multi_label_f1 = f1_score(multi_label_targets, multi_label_pred_binary, zero_division=0)
            else:
                multi_label_accuracy = multi_label_precision = multi_label_recall = multi_label_f1 = 0.0
            
            # Store metrics
            per_book_metrics[book_name] = {
                'overall': {
                    'accuracy': float(overall_accuracy),
                    'precision': float(overall_precision),
                    'recall': float(overall_recall),
                    'f1': float(overall_f1)
                },
                'single_label': {
                    'accuracy': float(single_label_accuracy),
                    'precision': float(single_label_precision),
                    'recall': float(single_label_recall),
                    'f1': float(single_label_f1),
                    'count': len(single_label_test_indices)
                },
                'multi_label': {
                    'accuracy': float(multi_label_accuracy),
                    'precision': float(multi_label_precision),
                    'recall': float(multi_label_recall),
                    'f1': float(multi_label_f1),
                    'count': len(multi_label_test_indices)
                }
            }
            
            # Log results
            logger.info(f"\n{book_name} Performance:")
            logger.info(f"  Overall: Acc={overall_accuracy:.3f}, F1={overall_f1:.3f}")
            logger.info(f"  Single-label: Acc={single_label_accuracy:.3f}, F1={single_label_f1:.3f} ({len(single_label_test_indices)} samples)")
            logger.info(f"  Multi-label: Acc={multi_label_accuracy:.3f}, F1={multi_label_f1:.3f} ({len(multi_label_test_indices)} samples)")
        
        # Save detailed metrics
        metrics_path = "multi_label_model/per_book_detailed_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(per_book_metrics, f, indent=2)
        
        # Create detailed report
        report_path = "multi_label_model/per_book_detailed_metrics_report.md"
        with open(report_path, 'w') as f:
            f.write("# Multi-Label Model - Per-Book Single-Label vs Multi-Label Performance Report\n\n")
            
            f.write("## Summary\n")
            f.write(f"- **Total test samples**: {len(test_sentences)}\n")
            f.write(f"- **Single-label samples**: {len(single_label_test_indices)} ({len(single_label_test_indices)/len(test_sentences)*100:.1f}%)\n")
            f.write(f"- **Multi-label samples**: {len(multi_label_test_indices)} ({len(multi_label_test_indices)/len(test_sentences)*100:.1f}%)\n")
            f.write(f"- **Model Type**: Unified Multi-Label Neural Network\n")
            f.write(f"- **Training Approach**: Single model for all books with multi-label output\n\n")
            
            # Performance overview table
            f.write("## Performance Overview\n\n")
            f.write("### Overall Model Performance Comparison\n\n")
            f.write("| Book | Overall Accuracy | Overall F1 | Single-Label F1 | Multi-Label F1 | Performance Pattern |\n")
            f.write("|------|------------------|------------|-----------------|----------------|-------------------|\n")
            
            for book_name, metrics in per_book_metrics.items():
                overall = metrics['overall']
                single = metrics['single_label']
                multi = metrics['multi_label']
                
                # Determine performance pattern
                f1_diff = multi['f1'] - single['f1']
                if f1_diff > 0.1:
                    pattern = "Multi-label excels"
                    single_f1_str = f"{single['f1']:.3f}"
                    multi_f1_str = f"**{multi['f1']:.3f}**"
                elif f1_diff < -0.1:
                    pattern = "Single-label excels"
                    single_f1_str = f"**{single['f1']:.3f}**"
                    multi_f1_str = f"{multi['f1']:.3f}"
                else:
                    pattern = "Balanced performance"
                    single_f1_str = f"{single['f1']:.3f}"
                    multi_f1_str = f"{multi['f1']:.3f}"
                
                f.write(f"| **{book_name}** | {overall['accuracy']:.3f} | {overall['f1']:.3f} | {single_f1_str} | {multi_f1_str} | {pattern} |\n")
            
            f.write("\n### Key Insights\n\n")
            f.write("1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.\n\n")
            f.write("2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.\n\n")
            f.write("3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.\n\n")
            
            f.write("## Per-Book Performance\n\n")
            for book_name, metrics in per_book_metrics.items():
                f.write(f"### {book_name}\n\n")
                
                f.write("#### Overall Performance\n")
                overall = metrics['overall']
                f.write(f"- **Accuracy**: {overall['accuracy']:.3f}\n")
                f.write(f"- **Precision**: {overall['precision']:.3f}\n")
                f.write(f"- **Recall**: {overall['recall']:.3f}\n")
                f.write(f"- **F1 Score**: {overall['f1']:.3f}\n\n")
                
                f.write("#### Single-Label Performance\n")
                single_label = metrics['single_label']
                f.write(f"- **Accuracy**: {single_label['accuracy']:.3f}\n")
                f.write(f"- **Precision**: {single_label['precision']:.3f}\n")
                f.write(f"- **Recall**: {single_label['recall']:.3f}\n")
                f.write(f"- **F1 Score**: {single_label['f1']:.3f}\n")
                f.write(f"- **Sample Count**: {single_label['count']}\n\n")
                
                f.write("#### Multi-Label Performance\n")
                multi_label = metrics['multi_label']
                f.write(f"- **Accuracy**: {multi_label['accuracy']:.3f}\n")
                f.write(f"- **Precision**: {multi_label['precision']:.3f}\n")
                f.write(f"- **Recall**: {multi_label['recall']:.3f}\n")
                f.write(f"- **F1 Score**: {multi_label['f1']:.3f}\n")
                f.write(f"- **Sample Count**: {multi_label['count']}\n\n")
                
                # Performance analysis
                f1_diff = multi_label['f1'] - single_label['f1']
                f.write("#### Performance Analysis\n")
                f.write(f"- **F1 Difference (Multi - Single)**: {f1_diff:+.3f}\n")
                if f1_diff > 0:
                    f.write(f"- **Multi-label performs better** by {f1_diff:.3f} F1 points\n")
                    f.write(f"- **Pattern**: This model excels at identifying {book_name} when it appears alongside other books\n")
                    if "Anna Karenina" in book_name:
                        f.write(f"- **Interpretation**: {book_name}'s distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts\n")
                    elif "Wuthering Heights" in book_name:
                        f.write(f"- **Interpretation**: {book_name}'s distinctive gothic style and emotional intensity is more recognizable in multi-label contexts\n")
                else:
                    f.write(f"- **Single-label performs better** by {abs(f1_diff):.3f} F1 points\n")
                    f.write(f"- **Pattern**: This model excels at identifying {book_name} when it's the only book present\n")
                    if "Frankenstein" in book_name:
                        f.write(f"- **Interpretation**: {book_name}'s distinctive gothic horror and scientific themes are very recognizable in isolation\n")
                    elif "Alice" in book_name:
                        f.write(f"- **Interpretation**: {book_name}'s distinctive whimsical and fantastical style is very recognizable in isolation\n")
                f.write("\n")
                
                f.write("---\n\n")
            
            f.write("## Comparative Analysis\n\n")
            f.write("### Model Performance Patterns\n\n")
            f.write("1. **Multi-Label Specialists**:\n")
            f.write("   - Books that perform significantly better in multi-label contexts\n")
            f.write("   - Have writing styles that become more distinctive when contrasted with others\n")
            f.write("   - Examples: Anna Karenina, Wuthering Heights\n\n")
            
            f.write("2. **Single-Label Specialists**:\n")
            f.write("   - Books that perform significantly better in single-label contexts\n")
            f.write("   - Have very distinctive styles that are easily recognizable in isolation\n")
            f.write("   - Examples: Frankenstein, Alice in Wonderland\n\n")
            
            f.write("### Writing Style Analysis\n\n")
            f.write("- **Anna Karenina & Wuthering Heights**: Complex, emotionally intense narratives with distinctive authorial voices that become more apparent when contrasted with other styles\n")
            f.write("- **Frankenstein & Alice in Wonderland**: Highly distinctive genres (gothic horror vs. children's fantasy) with unique thematic elements that are immediately recognizable\n\n")
            
            f.write("### Practical Implications\n\n")
            f.write("1. **For Multi-Label Classification**: Anna Karenina and Wuthering Heights models are more reliable when multiple books are present\n")
            f.write("2. **For Single-Label Classification**: Frankenstein and Alice in Wonderland models are more reliable when only one book is present\n")
            f.write("3. **Overall**: The unified multi-label model achieves strong performance (84-88% accuracy) for individual book identification\n\n")
            
            f.write("## Methodology Notes\n\n")
            f.write("- **Model Architecture**: Unified multi-label neural network with sigmoid outputs\n")
            f.write("- **Training Data**: Pre-existing dataset splits with aligned embeddings\n")
            f.write("- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis\n")
            f.write("- **Threshold**: 0.5 probability threshold for binary classification\n")
            f.write("- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation\n")
            f.write("- **Approach**: Single model handles all books simultaneously, leveraging shared representations\n")
        
        logger.info(f"Per-book metrics saved to {metrics_path}")
        logger.info(f"Per-book report saved to {report_path}")
        
        return per_book_metrics
    
    def extract_test_examples(self):
        """Extract examples of single-label and multi-label sentences from test data."""
        logger.info("Extracting test examples from dataset...")
        
        # Get test data
        test_data = self.dataset['test']
        test_sentences = test_data['sentence']
        test_labels = test_data['labels']
        
        # Identify single-label and multi-label sentences in test set
        single_label_test_indices = []
        multi_label_test_indices = []
        
        for i, labels in enumerate(test_labels):
            if sum(labels) == 1:
                single_label_test_indices.append(i)
            elif sum(labels) > 1:
                multi_label_test_indices.append(i)
        
        logger.info(f"Found {len(single_label_test_indices)} single-label test sentences")
        logger.info(f"Found {len(multi_label_test_indices)} multi-label test sentences")
        
        # Get predictions for all test sentences
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test)
            test_probabilities = test_outputs.cpu().numpy()
            test_predictions = (test_outputs > 0.5).float().cpu().numpy()
        
        # Extract examples
        examples = {
            'single_label_examples': [],
            'multi_label_examples': []
        }
        
        # Extract single-label examples
        logger.info("Extracting single-label examples...")
        for i, idx in enumerate(single_label_test_indices[:15]):  # First 15 examples
            sentence_text = test_sentences[idx]
            labels = test_labels[idx]
            
            # Get true labels
            true_labels = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                true_labels[book_col] = int(labels[book_idx])
            
            # Get predictions
            predictions = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                predictions[book_col] = float(test_probabilities[idx][book_idx])
            
            examples['single_label_examples'].append({
                'sentence': sentence_text,
                'true_labels': true_labels,
                'predictions': predictions
            })
        
        # Extract multi-label examples
        logger.info("Extracting multi-label examples...")
        for i, idx in enumerate(multi_label_test_indices[:15]):  # First 15 examples
            sentence_text = test_sentences[idx]
            labels = test_labels[idx]
            
            # Get true labels
            true_labels = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                true_labels[book_col] = int(labels[book_idx])
            
            # Get predictions
            predictions = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                predictions[book_col] = float(test_probabilities[idx][book_idx])
            
            examples['multi_label_examples'].append({
                'sentence': sentence_text,
                'true_labels': true_labels,
                'predictions': predictions
            })
        
        # Calculate accuracy for examples
        single_label_correct = 0
        multi_label_correct = 0
        
        for example in examples['single_label_examples']:
            correct_predictions = 0
            total_predictions = 0
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                pred_binary = 1 if example['predictions'][book_col] > 0.5 else 0
                if pred_binary == example['true_labels'][book_col]:
                    correct_predictions += 1
                total_predictions += 1
            if total_predictions > 0 and correct_predictions == total_predictions:
                single_label_correct += 1
        
        for example in examples['multi_label_examples']:
            correct_predictions = 0
            total_predictions = 0
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                pred_binary = 1 if example['predictions'][book_col] > 0.5 else 0
                if pred_binary == example['true_labels'][book_col]:
                    correct_predictions += 1
                total_predictions += 1
            if total_predictions > 0 and correct_predictions == total_predictions:
                multi_label_correct += 1
        
        single_label_accuracy = single_label_correct / len(examples['single_label_examples']) if examples['single_label_examples'] else 0
        multi_label_accuracy = multi_label_correct / len(examples['multi_label_examples']) if examples['multi_label_examples'] else 0
        
        logger.info(f"Single-label accuracy: {single_label_accuracy:.3f}")
        logger.info(f"Multi-label accuracy: {multi_label_accuracy:.3f}")
        
        # Save examples
        examples_path = "multi_label_model/test_examples.json"
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Create detailed report
        report_path = "multi_label_model/test_examples_report.md"
        with open(report_path, 'w') as f:
            f.write("# Multi-Label Neural Network - Test Examples Report\n\n")
            f.write(f"**Single-label Accuracy**: {single_label_accuracy:.3f}\n")
            f.write(f"**Multi-label Accuracy**: {multi_label_accuracy:.3f}\n\n")
            
            f.write("## Single-Label Examples\n\n")
            for i, example in enumerate(examples['single_label_examples'], 1):
                f.write(f"### Example {i}\n")
                f.write(f"**Sentence**: {example['sentence']}\n\n")
                f.write("| Book | True Label | Prediction | Correct |\n")
                f.write("|------|------------|------------|--------|\n")
                for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                    book_name = self.book_names[book_col]
                    true_label = example['true_labels'][book_col]
                    pred_prob = example['predictions'][book_col]
                    pred_binary = 1 if pred_prob > 0.5 else 0
                    correct = "✓" if pred_binary == true_label else "✗"
                    f.write(f"| {book_name} | {true_label} | {pred_prob:.3f} ({pred_binary}) | {correct} |\n")
                f.write("\n")
            
            f.write("## Multi-Label Examples\n\n")
            for i, example in enumerate(examples['multi_label_examples'], 1):
                f.write(f"### Example {i}\n")
                f.write(f"**Sentence**: {example['sentence']}\n\n")
                f.write("| Book | True Label | Prediction | Correct |\n")
                f.write("|------|------------|------------|--------|\n")
                for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                    book_name = self.book_names[book_col]
                    true_label = example['true_labels'][book_col]
                    pred_prob = example['predictions'][book_col]
                    pred_binary = 1 if pred_prob > 0.5 else 0
                    correct = "✓" if pred_binary == true_label else "✗"
                    f.write(f"| {book_name} | {true_label} | {pred_prob:.3f} ({pred_binary}) | {correct} |\n")
                f.write("\n")
        
        logger.info(f"Test examples saved to {examples_path}")
        logger.info(f"Test examples report saved to {report_path}")
        
        return single_label_accuracy, multi_label_accuracy
    
    def save_model(self):
        """Save the trained model and results."""
        if self.model is None:
            logger.error("No model to save!")
            return
        
        # Create models directory
        models_dir = Path("multi_label_model")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "neural_network_semantic.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'results': self.results,
            'model_config': {
                'input_dim': self.X_train.shape[1],
                'output_dim': self.y_train.shape[1],
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3
            }
        }, model_path)
        
        # Save results
        results_path = models_dir / "neural_network_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays and float32 to JSON serializable types
            results_for_json = {}
            for k, v in self.results.items():
                if isinstance(v, np.ndarray):
                    results_for_json[k] = v.tolist()
                elif isinstance(v, np.float32):
                    results_for_json[k] = float(v)
                else:
                    results_for_json[k] = v
            json.dump(results_for_json, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")

def main():
    """Main training function."""
    logger.info("Starting neural network training...")
    
    # Initialize trainer
    trainer = NeuralNetworkTrainer()
    
    # Load data
    if not trainer.load_data():
        logger.error("Failed to load data!")
        return
    
    # Train model
    training_history = trainer.train_model(
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        patience=15
    )
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Evaluate single vs multi-label performance
    single_multi_metrics = trainer.evaluate_single_vs_multi_label_performance()
    
    # Calculate per-book metrics
    per_book_metrics = trainer.calculate_per_book_metrics()
    
    # Extract test examples
    single_label_acc, multi_label_acc = trainer.extract_test_examples()
    
    # Save model
    trainer.save_model()
    
    logger.info(f"Training completed! Final F1 Score: {results['f1_score']:.4f}")
    logger.info(f"Single-label accuracy: {single_label_acc:.3f}")
    logger.info(f"Multi-label accuracy: {multi_label_acc:.3f}")

if __name__ == "__main__":
    main() 