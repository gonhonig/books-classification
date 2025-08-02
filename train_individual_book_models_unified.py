"""
Unified Individual Book Models Training
Creates a single balanced dataset for all book models, prioritizing multi-label sentences
and balancing single-label sentences across books. Uses the same dataset for all models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

class UnifiedIndividualBookTrainer:
    """Trainer for individual book models using a unified balanced dataset."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Unified dataset properties
        self.X_unified = None
        self.y_unified = {}
        self.selected_indices = None
        
        # Train/val/test splits for each book
        self.train_splits = {}
        self.val_splits = {}
        self.test_splits = {}
        
        # Book names mapping
        self.book_names = {
            'book_1': 'Anna Karenina',
            'book_2': 'Wuthering Heights', 
            'book_3': 'Frankenstein',
            'book_4': 'The Adventures of Alice in Wonderland'
        }
        
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
        return df
    
    def create_unified_balanced_dataset(self, target_samples_per_class=5000):
        """Create a unified balanced dataset for all books, prioritizing multi-label sentences."""
        logger.info("Creating unified balanced dataset for all books...")
        
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
        # We want equal numbers of positive and negative samples for each book
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
        
        return self.X_unified, self.y_unified, self.selected_indices
    
    def create_train_val_test_splits(self):
        """Create train/val/test splits for all books once and store them."""
        logger.info("Creating train/val/test splits for all books...")
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            y_book = self.y_unified[book_col]
            
            # Use the same random seed as training
            np.random.seed(42)
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X_unified, y_book, test_size=0.3, random_state=42, stratify=y_book
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # Store splits
            self.train_splits[book_col] = {
                'X': X_train,
                'y': y_train
            }
            self.val_splits[book_col] = {
                'X': X_val,
                'y': y_val
            }
            self.test_splits[book_col] = {
                'X': X_test,
                'y': y_test
            }
            
            logger.info(f"Created splits for {self.book_names[book_col]}: "
                       f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    def train_book_model(self, book_col, X_unified, y_unified, hidden_dims=[256, 128, 64], dropout_rate=0.3,
                        epochs=100, batch_size=32, learning_rate=0.001, patience=15):
        """Train a model for a specific book using the unified dataset."""
        book_name = self.book_names[book_col]
        logger.info(f"Training model for {book_name}")
        
        # Get stored splits for this book
        train_data = self.train_splits[book_col]
        val_data = self.val_splits[book_col]
        test_data = self.test_splits[book_col]
        
        X_train, y_train = train_data['X'], train_data['y']
        X_val, y_val = val_data['X'], val_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
                torch.save(model.state_dict(), f"models/{book_name.replace(' ', '_').lower()}_best_model.pth")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f"models/{book_name.replace(' ', '_').lower()}_best_model.pth"))
        
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
        multi_label_metrics = self.evaluate_multi_label_performance(model, scaler, X_test_scaled, y_test, book_col)
        
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
        """Train models for all books using the unified dataset."""
        logger.info("Training models for all books using unified dataset...")
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
        # Create unified balanced dataset (stored as properties)
        self.create_unified_balanced_dataset()
        
        # Create train/val/test splits once
        self.create_train_val_test_splits()
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            try:
                self.train_book_model(book_col, self.X_unified, self.y_unified)
            except Exception as e:
                logger.error(f"Error training model for {book_col}: {e}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training results."""
        results_path = "models/unified_individual_book_results.json"
        
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
        fig.suptitle('Unified Individual Book Model Performance Comparison', fontsize=16)
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            book_names = []
            metric_values = []
            
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
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
        plt.savefig('models/unified_individual_book_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comparison plot saved to models/unified_individual_book_model_comparison.png")

    def evaluate_multi_label_performance(self, model, scaler, X_test, y_test, book_col):
        """Evaluate model performance specifically on multi-label sentences in test set."""
        book_name = self.book_names[book_col]
        
        # We need to identify multi-label sentences in the test set
        # Use the stored test split for this book
        test_data = self.test_splits[book_col]
        X_test_full = test_data['X']
        y_test_full = test_data['y']
        
        # Get the test indices in the unified dataset
        test_size = len(X_test_full)
        test_indices_in_unified = list(range(len(self.X_unified) - test_size, len(self.X_unified)))
        
        # Convert unified dataset indices back to original dataset indices
        original_test_indices = [self.selected_indices[i] for i in test_indices_in_unified]
        
        # Identify multi-label sentences in the test set
        test_multi_label_indices = []
        for i, original_idx in enumerate(original_test_indices):
            # Get all book labels for this sentence
            book_labels_for_sentence = [
                self.book_labels['book_1'][original_idx],
                self.book_labels['book_2'][original_idx], 
                self.book_labels['book_3'][original_idx],
                self.book_labels['book_4'][original_idx]
            ]
            # Check if this is a multi-label sentence (belongs to more than one book)
            if sum(book_labels_for_sentence) > 1:
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
        logger.info("Extracting test examples from unified dataset...")
        
        # Load the original dataset to get sentence text
        df = pd.read_csv(self.data_path)
        
        # Load trained models
        models = {}
        scalers = {}
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            book_name = self.book_names[book_col]
            model_path = f"models/{book_name.replace(' ', '_').lower()}_best_model.pth"
            
            if Path(model_path).exists():
                model = BinaryBookClassifier().to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                models[book_col] = model
                scalers[book_col] = self.scalers[book_col]
                logger.info(f"Loaded model for {book_name}")
        
        # Get test indices from the unified dataset
        # Use the stored test splits for all books
        test_indices_all_books = set()
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            test_data = self.test_splits[book_col]
            X_test = test_data['X']
            
            # Get test indices in the unified dataset
            test_size = len(X_test)
            test_indices = list(range(len(self.X_unified) - test_size, len(self.X_unified)))
            test_indices_all_books.update(test_indices)
        
        # Convert unified dataset indices back to original dataset indices
        original_test_indices = [self.selected_indices[i] for i in test_indices_all_books]
        
        # Identify single-label and multi-label sentences in test set
        single_label_test_indices = []
        multi_label_test_indices = []
        
        for idx in original_test_indices:
            book_labels_for_sentence = [
                self.book_labels['book_1'][idx],
                self.book_labels['book_2'][idx], 
                self.book_labels['book_3'][idx],
                self.book_labels['book_4'][idx]
            ]
            
            if sum(book_labels_for_sentence) == 1:
                single_label_test_indices.append(idx)
            elif sum(book_labels_for_sentence) > 1:
                multi_label_test_indices.append(idx)
        
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
            sentence_text = df.iloc[idx]['sentence'] if 'sentence' in df.columns else f"Sentence {idx}"
            
            # Get true labels
            true_labels = {
                'book_1': int(self.book_labels['book_1'][idx]),
                'book_2': int(self.book_labels['book_2'][idx]),
                'book_3': int(self.book_labels['book_3'][idx]),
                'book_4': int(self.book_labels['book_4'][idx])
            }
            
            # Get predictions from all models
            predictions = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                if book_col in models:
                    model = models[book_col]
                    scaler = scalers[book_col]
                    
                    # Get embedding for this sentence
                    embedding = self.embeddings[idx:idx+1]
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
            sentence_text = df.iloc[idx]['sentence'] if 'sentence' in df.columns else f"Sentence {idx}"
            
            # Get true labels
            true_labels = {
                'book_1': int(self.book_labels['book_1'][idx]),
                'book_2': int(self.book_labels['book_2'][idx]),
                'book_3': int(self.book_labels['book_3'][idx]),
                'book_4': int(self.book_labels['book_4'][idx])
            }
            
            # Get predictions from all models
            predictions = {}
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
                if book_col in models:
                    model = models[book_col]
                    scaler = scalers[book_col]
                    
                    # Get embedding for this sentence
                    embedding = self.embeddings[idx:idx+1]
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
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
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
            for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
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
        examples_path = "models/unified_test_examples.json"
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Create detailed report
        report_path = "models/unified_test_examples_report.md"
        with open(report_path, 'w') as f:
            f.write("# Unified Individual Book Models - Test Examples Report\n\n")
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

def main():
    """Main training function."""
    logger.info("Starting unified individual book model training...")
    
    # Initialize trainer
    trainer = UnifiedIndividualBookTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Create comparison plot
    trainer.create_comparison_plot()
    
    # Extract test examples
    logger.info("Extracting test examples...")
    single_label_acc, multi_label_acc = trainer.extract_test_examples()
    
    logger.info("=" * 60)
    logger.info("UNIFIED TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Single-label accuracy: {single_label_acc:.3f}")
    logger.info(f"Multi-label accuracy: {multi_label_acc:.3f}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 