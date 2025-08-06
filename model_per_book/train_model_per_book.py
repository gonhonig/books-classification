"""
Individual Book Models Training
Uses pre-existing dataset splits from data/dataset for training individual book models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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

class IndividualBookTrainer:
    """Trainer for individual book models using pre-existing dataset splits."""
    
    def __init__(self, dataset_path: str = "data/dataset", embeddings_path: str = "data/embeddings_cache_aligned_f24a423ed8f9dd531230fe64f71f668d.npz", config_path: str = "configs/optimized_params_config.yaml"):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.config = self.load_config()
        
        # Dataset properties
        self.dataset = None
        self.embeddings = None
        self.book_labels = {}
        
        # Load book-to-label mapping from JSON file
        self.book_mapping = self.load_book_mapping()
        
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

        # Book order in labels list (from the mapping file)
        self.book_order = self.book_mapping['books']
        
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
        
        # Extract book labels from the labels column
        self.book_labels = {}
        for book_col in self.book_columns:
            book_name = self.book_names[book_col]
            book_idx = self.book_order.index(book_name)
            
            # Extract labels from all splits
            train_labels = [labels[book_idx] for labels in self.dataset['train']['labels']]
            val_labels = [labels[book_idx] for labels in self.dataset['validation']['labels']]
            test_labels = [labels[book_idx] for labels in self.dataset['test']['labels']]
            
            # Combine all labels
            all_labels = np.concatenate([train_labels, val_labels, test_labels])
            self.book_labels[book_col] = all_labels.astype(np.float32)
            
            positive_count = np.sum(all_labels == 1)
            logger.info(f"Loaded {book_col} ({book_name}): {positive_count} positive samples")
        
        logger.info("Data loaded successfully")
        
    def get_split_data(self, split_name: str, book_col: str):
        """Get data for a specific split and book."""
        split_data = self.dataset[split_name]
        book_name = self.book_names[book_col]
        book_idx = self.book_order.index(book_name)
        
        # Get sentences and labels
        sentences = split_data['sentence']
        labels = [labels_list[book_idx] for labels_list in split_data['labels']]
        labels = np.array(labels, dtype=np.float32)
        
        # Get corresponding embeddings
        split_size = len(sentences)
        if split_name == 'train':
            start_idx = 0
        elif split_name == 'validation':
            start_idx = len(self.dataset['train'])
        else:  # test
            start_idx = len(self.dataset['train']) + len(self.dataset['validation'])
        
        end_idx = start_idx + split_size
        embeddings = self.embeddings[start_idx:end_idx]
        
        return embeddings, labels, sentences
    
    def train_book_model(self, book_col, hidden_dims=None, dropout_rate=None,
                        epochs=None, batch_size=None, learning_rate=None, patience=None, weight_decay=None):
        """Train a model for a specific book using the pre-existing splits."""
        book_name = self.book_names[book_col]
        logger.info(f"Training model for {book_name}")
        
        # Use config parameters if available, otherwise use defaults
        if self.config and 'per_book' in self.config and book_col in self.config['per_book']:
            config_params = self.config['per_book'][book_col]
            hidden_dims = hidden_dims or config_params['architecture']['hidden_dims']
            dropout_rate = dropout_rate or config_params['architecture']['dropout_rate']
            epochs = epochs or config_params['training']['epochs']
            batch_size = batch_size or config_params['training']['batch_size']
            learning_rate = learning_rate or config_params['training']['learning_rate']
            patience = patience or config_params['training']['patience']
            weight_decay = weight_decay or config_params['training']['weight_decay']
        else:
            hidden_dims = hidden_dims or [256, 128, 64]
            dropout_rate = dropout_rate or 0.3
            epochs = epochs or 100
            batch_size = batch_size or 32
            learning_rate = learning_rate or 0.001
            patience = patience or 15
            weight_decay = weight_decay or 1e-5
        
        logger.info(f"Training parameters for {book_name}: hidden_dims={hidden_dims}, dropout_rate={dropout_rate}")
        logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, patience={patience}, weight_decay={weight_decay}")
        
        # Get data for each split
        X_train, y_train, train_sentences = self.get_split_data('train', book_col)
        X_val, y_val, val_sentences = self.get_split_data('validation', book_col)
        X_test, y_test, test_sentences = self.get_split_data('test', book_col)
        
        logger.info(f"Train: {len(X_train)} samples, {np.sum(y_train)} positive")
        logger.info(f"Validation: {len(X_val)} samples, {np.sum(y_val)} positive")
        logger.info(f"Test: {len(X_test)} samples, {np.sum(y_test)} positive")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = BinaryBookClassifier(
            input_dim=384,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
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
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"model_per_book/{book_name.replace(' ', '_').lower()}_best_model.pth")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f"model_per_book/{book_name.replace(' ', '_').lower()}_best_model.pth"))
        
        # Test evaluation
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        # Convert to numpy arrays
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # Convert probabilities to binary predictions
        test_pred_binary = (test_predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(test_targets, test_pred_binary)
        precision = precision_score(test_targets, test_pred_binary)
        recall = recall_score(test_targets, test_pred_binary)
        f1 = f1_score(test_targets, test_pred_binary)
        
        # Evaluate multi-label performance
        multi_label_metrics = self.evaluate_multi_label_performance(model, scaler, X_test_scaled, y_test, book_col, test_sentences)
        
        # Store results
        self.models[book_col] = model
        self.scalers[book_col] = scaler
        self.results[book_col] = {
            'book_name': book_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_predictions': test_predictions,
            'test_targets': test_targets,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'multi_label_metrics': multi_label_metrics
        }
        
        logger.info(f"{book_name} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return model, scaler, self.results[book_col]
    
    def train_all_models(self):
        """Train models for all books using the pre-existing dataset splits."""
        logger.info("Training models for all books using pre-existing dataset splits...")
        
        # Create models directory if it doesn't exist
        Path("model_per_book").mkdir(exist_ok=True)
        
        for book_col in self.book_columns:
            try:
                self.train_book_model(book_col)
            except Exception as e:
                logger.error(f"Error training model for {book_col}: {e}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training results."""
        results_path = "model_per_book/results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for book_col, result in self.results.items():
            serializable_results[book_col] = {
                'book_name': result['book_name'],
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'test_predictions': result['test_predictions'].tolist(),
                'test_targets': result['test_targets'].tolist(),
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses']
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def create_comparison_plot(self):
        """Create comparison plot of all models."""
        if not self.results:
            logger.warning("No results available for plotting")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Individual Book Model Performance Comparison', fontsize=16)
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            book_names = []
            metric_values = []
            
            for book_col in self.book_columns:
                if book_col in self.results:
                    book_names.append(self.results[book_col]['book_name'])
                    metric_values.append(self.results[book_col][metric])
            
            bars = ax.bar(book_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_per_book/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comparison plot saved to model_per_book/model_comparison.png")

    def evaluate_multi_label_performance(self, model, scaler, X_test, y_test, book_col, test_sentences):
        """Evaluate model performance specifically on multi-label sentences in test set."""
        book_name = self.book_names[book_col]
        
        # Identify multi-label sentences in the test set
        test_multi_label_indices = []
        for i, sentence in enumerate(test_sentences):
            # Get all book labels for this sentence from the test set
            labels = self.dataset['test']['labels'][i]
            
            # Check if this is a multi-label sentence (belongs to more than one book)
            if sum(labels) > 1:
                test_multi_label_indices.append(i)
        
        if len(test_multi_label_indices) == 0:
            logger.info(f"No multi-label sentences found in test set for {book_name}")
            return {
                'multi_label_accuracy': 0.0,
                'multi_label_precision': 0.0,
                'multi_label_recall': 0.0,
                'multi_label_f1': 0.0,
                'multi_label_count': 0
            }
        
        # Get predictions for multi-label sentences only
        model.eval()
        multi_label_predictions = []
        multi_label_targets = []
        
        with torch.no_grad():
            for idx in test_multi_label_indices:
                X_sample = torch.FloatTensor(X_test[idx:idx+1]).to(self.device)
                output = model(X_sample)
                if output.dim() == 0:
                    pred = output.cpu().numpy()
                else:
                    pred = output.cpu().numpy()[0]
                multi_label_predictions.append(pred)
                multi_label_targets.append(y_test[idx])
        
        # Convert to binary predictions
        multi_label_pred_binary = (np.array(multi_label_predictions) > 0.5).astype(int)
        multi_label_targets = np.array(multi_label_targets)
        
        # Calculate metrics
        multi_label_accuracy = accuracy_score(multi_label_targets, multi_label_pred_binary)
        multi_label_precision = precision_score(multi_label_targets, multi_label_pred_binary)
        multi_label_recall = recall_score(multi_label_targets, multi_label_pred_binary)
        multi_label_f1 = f1_score(multi_label_targets, multi_label_pred_binary)
        
        logger.info(f"{book_name} Multi-label Performance:")
        logger.info(f"Multi-label Accuracy: {multi_label_accuracy:.4f}")
        logger.info(f"Multi-label Precision: {multi_label_precision:.4f}")
        logger.info(f"Multi-label Recall: {multi_label_recall:.4f}")
        logger.info(f"Multi-label F1: {multi_label_f1:.4f}")
        logger.info(f"Multi-label Count: {len(test_multi_label_indices)}")
        
        return {
            'multi_label_accuracy': multi_label_accuracy,
            'multi_label_precision': multi_label_precision,
            'multi_label_recall': multi_label_recall,
            'multi_label_f1': multi_label_f1,
            'multi_label_count': len(test_multi_label_indices)
        }
    
    def extract_test_examples(self):
        """Extract examples of single-label and multi-label sentences from test data."""
        logger.info("Extracting test examples from dataset...")
        
        # Load trained models
        models = {}
        scalers = {}
        for book_col in self.book_columns:
            book_name = self.book_names[book_col]
            model_path = f"model_per_book/{book_name.replace(' ', '_').lower()}_best_model.pth"
            
            if Path(model_path).exists():
                # Get the correct architecture parameters from config
                if self.config and 'per_book' in self.config and book_col in self.config['per_book']:
                    config_params = self.config['per_book'][book_col]
                    hidden_dims = config_params['architecture']['hidden_dims']
                    dropout_rate = config_params['architecture']['dropout_rate']
                else:
                    # Fallback to default parameters
                    hidden_dims = [256, 128, 64]
                    dropout_rate = 0.3
                
                model = BinaryBookClassifier(
                    input_dim=384,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                models[book_col] = model
                scalers[book_col] = self.scalers[book_col]
                logger.info(f"Loaded model for {book_name} with architecture: hidden_dims={hidden_dims}, dropout_rate={dropout_rate}")
        
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
            for book_col in self.book_columns:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                true_labels[book_col] = int(labels[book_idx])
            
            # Get predictions from all models
            predictions = {}
            for book_col in self.book_columns:
                if book_col in models:
                    model = models[book_col]
                    scaler = scalers[book_col]
                    
                    # Get embedding for this sentence
                    start_idx = len(self.dataset['train']) + len(self.dataset['validation'])
                    embedding = self.embeddings[start_idx + idx:start_idx + idx + 1]
                    embedding_scaled = scaler.transform(embedding)
                    
                    with torch.no_grad():
                        output = model(torch.FloatTensor(embedding_scaled).to(self.device))
                        if output.dim() == 0:
                            pred = output.cpu().numpy()
                        else:
                            pred = output.cpu().numpy()[0]
                        predictions[book_col] = float(pred)
            
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
            for book_col in self.book_columns:
                book_name = self.book_names[book_col]
                book_idx = self.book_order.index(book_name)
                true_labels[book_col] = int(labels[book_idx])
            
            # Get predictions from all models
            predictions = {}
            for book_col in self.book_columns:
                if book_col in models:
                    model = models[book_col]
                    scaler = scalers[book_col]
                    
                    # Get embedding for this sentence
                    start_idx = len(self.dataset['train']) + len(self.dataset['validation'])
                    embedding = self.embeddings[start_idx + idx:start_idx + idx + 1]
                    embedding_scaled = scaler.transform(embedding)
                    
                    with torch.no_grad():
                        output = model(torch.FloatTensor(embedding_scaled).to(self.device))
                        if output.dim() == 0:
                            pred = output.cpu().numpy()
                        else:
                            pred = output.cpu().numpy()[0]
                        predictions[book_col] = float(pred)
            
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
            for book_col in self.book_columns:
                if book_col in example['predictions']:
                    pred_binary = 1 if example['predictions'][book_col] > 0.5 else 0
                    if pred_binary == example['true_labels'][book_col]:
                        correct_predictions += 1
                    total_predictions += 1
            if total_predictions > 0 and correct_predictions == total_predictions:
                single_label_correct += 1
        
        for example in examples['multi_label_examples']:
            correct_predictions = 0
            total_predictions = 0
            for book_col in self.book_columns:
                if book_col in example['predictions']:
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
        examples_path = "model_per_book/per_book_test_examples.json"
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Create detailed report
        report_path = "model_per_book/per_book_test_examples_report.md"
        with open(report_path, 'w') as f:
            f.write("# Individual Book Models - Test Examples Report\n\n")
            f.write(f"**Single-label Accuracy**: {single_label_accuracy:.3f}\n")
            f.write(f"**Multi-label Accuracy**: {multi_label_accuracy:.3f}\n\n")
            
            f.write("## Single-Label Examples\n\n")
            for i, example in enumerate(examples['single_label_examples'], 1):
                f.write(f"### Example {i}\n")
                f.write(f"**Sentence**: {example['sentence']}\n\n")
                f.write("| Book | True Label | Prediction | Correct |\n")
                f.write("|------|------------|------------|--------|\n")
                for book_col in self.book_columns:
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
                for book_col in self.book_columns:
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

    def extract_per_book_metrics(self):
        """Extract detailed per-book metrics for single-label vs multi-label performance."""
        logger.info("Extracting per-book single-label vs multi-label metrics...")
        
        # Load trained models
        models = {}
        scalers = {}
        for book_col in self.book_columns:
            book_name = self.book_names[book_col]
            model_path = f"model_per_book/{book_name.replace(' ', '_').lower()}_best_model.pth"
            
            if Path(model_path).exists():
                # Get the correct architecture parameters from config
                if self.config and 'per_book' in self.config and book_col in self.config['per_book']:
                    config_params = self.config['per_book'][book_col]
                    hidden_dims = config_params['architecture']['hidden_dims']
                    dropout_rate = config_params['architecture']['dropout_rate']
                else:
                    # Fallback to default parameters
                    hidden_dims = [256, 128, 64]
                    dropout_rate = 0.3
                
                model = BinaryBookClassifier(
                    input_dim=384,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                models[book_col] = model
                scalers[book_col] = self.scalers[book_col]
                logger.info(f"Loaded model for {book_name} with architecture: hidden_dims={hidden_dims}, dropout_rate={dropout_rate}")
        
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
        
        # Calculate per-book metrics
        per_book_metrics = {}
        
        for book_col in self.book_columns:
            book_name = self.book_names[book_col]
            book_idx = self.book_order.index(book_name)
            
            if book_col not in models:
                continue
                
            model = models[book_col]
            scaler = scalers[book_col]
            
            # Get predictions for all test sentences
            all_predictions = []
            all_targets = []
            
            for i in range(len(test_sentences)):
                # Get embedding for this sentence
                start_idx = len(self.dataset['train']) + len(self.dataset['validation'])
                embedding = self.embeddings[start_idx + i:start_idx + i + 1]
                embedding_scaled = scaler.transform(embedding)
                
                with torch.no_grad():
                    output = model(torch.FloatTensor(embedding_scaled).to(self.device))
                    if output.dim() == 0:
                        pred = output.cpu().numpy()
                    else:
                        pred = output.cpu().numpy()[0]
                    all_predictions.append(float(pred))
                    all_targets.append(int(test_labels[i][book_idx]))
            
            # Convert to binary predictions
            all_pred_binary = (np.array(all_predictions) > 0.5).astype(int)
            all_targets = np.array(all_targets)
            
            # Calculate overall metrics
            overall_accuracy = accuracy_score(all_targets, all_pred_binary)
            overall_precision = precision_score(all_targets, all_pred_binary)
            overall_recall = recall_score(all_targets, all_pred_binary)
            overall_f1 = f1_score(all_targets, all_pred_binary)
            
            # Calculate single-label metrics
            single_label_predictions = []
            single_label_targets = []
            for idx in single_label_test_indices:
                single_label_predictions.append(all_predictions[idx])
                single_label_targets.append(all_targets[idx])
            
            if len(single_label_predictions) > 0:
                single_label_pred_binary = (np.array(single_label_predictions) > 0.5).astype(int)
                single_label_targets = np.array(single_label_targets)
                
                single_label_accuracy = accuracy_score(single_label_targets, single_label_pred_binary)
                single_label_precision = precision_score(single_label_targets, single_label_pred_binary)
                single_label_recall = recall_score(single_label_targets, single_label_pred_binary)
                single_label_f1 = f1_score(single_label_targets, single_label_pred_binary)
            else:
                single_label_accuracy = single_label_precision = single_label_recall = single_label_f1 = 0.0
            
            # Calculate multi-label metrics
            multi_label_predictions = []
            multi_label_targets = []
            for idx in multi_label_test_indices:
                multi_label_predictions.append(all_predictions[idx])
                multi_label_targets.append(all_targets[idx])
            
            if len(multi_label_predictions) > 0:
                multi_label_pred_binary = (np.array(multi_label_predictions) > 0.5).astype(int)
                multi_label_targets = np.array(multi_label_targets)
                
                multi_label_accuracy = accuracy_score(multi_label_targets, multi_label_pred_binary)
                multi_label_precision = precision_score(multi_label_targets, multi_label_pred_binary)
                multi_label_recall = recall_score(multi_label_targets, multi_label_pred_binary)
                multi_label_f1 = f1_score(multi_label_targets, multi_label_pred_binary)
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
        metrics_path = "model_per_book/per_book_detailed_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(per_book_metrics, f, indent=2)
        
        # Create detailed report
        report_path = "model_per_book/per_book_detailed_metrics_report.md"
        with open(report_path, 'w') as f:
            f.write("# Individual Book Models - Per-Book Single-Label vs Multi-Label Performance Report\n\n")
            
            f.write("## Summary\n")
            f.write(f"- **Total test samples**: {len(test_sentences)}\n")
            f.write(f"- **Single-label samples**: {len(single_label_test_indices)} ({len(single_label_test_indices)/len(test_sentences)*100:.1f}%)\n")
            f.write(f"- **Multi-label samples**: {len(multi_label_test_indices)} ({len(multi_label_test_indices)/len(test_sentences)*100:.1f}%)\n")
            f.write(f"- **Model Type**: Individual Binary Classifiers (one model per book)\n")
            f.write(f"- **Training Approach**: Separate binary classification for each book\n\n")
            
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
                
                f.write("---\n\n")
        
        logger.info(f"Detailed metrics saved to {metrics_path}")
        logger.info(f"Detailed report saved to {report_path}")
        
        return per_book_metrics

def main():
    """Main training function."""
    logger.info("Starting individual book model training...")
    
    # Initialize trainer
    trainer = IndividualBookTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Create comparison plot
    trainer.create_comparison_plot()
    
    # Extract test examples
    logger.info("Extracting test examples...")
    single_label_acc, multi_label_acc = trainer.extract_test_examples()
    
    # Extract per-book metrics
    trainer.extract_per_book_metrics()
    
    logger.info("=" * 60)
    logger.info("INDIVIDUAL BOOK TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Single-label accuracy: {single_label_acc:.3f}")
    logger.info(f"Multi-label accuracy: {multi_label_acc:.3f}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 