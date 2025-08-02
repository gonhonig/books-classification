"""
Train Individual Book Models
Trains 4 separate neural networks, one for each book, using the semantic augmented dataset.
Each model performs binary classification (belongs to book or not).
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

class IndividualBookTrainer:
    """Trainer for individual book models."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.results = {}
        
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
    
    def create_balanced_dataset(self, book_col, target_samples_per_class=5000):
        """Create a balanced dataset for a specific book while preserving multi-label sentences."""
        logger.info(f"Creating balanced dataset for {self.book_names[book_col]}")
        
        # Get positive and negative samples
        positive_indices = np.where(self.book_labels[book_col] == 1)[0]
        negative_indices = np.where(self.book_labels[book_col] == 0)[0]
        
        logger.info(f"Found {len(positive_indices)} positive and {len(negative_indices)} negative samples")
        
        # Identify multi-label sentences (sentences that belong to multiple books)
        multi_label_indices = []
        for i in range(len(self.embeddings)):
            # Count how many books this sentence belongs to
            book_labels_for_sentence = [
                self.book_labels['book_1'][i],
                self.book_labels['book_2'][i], 
                self.book_labels['book_3'][i],
                self.book_labels['book_4'][i]
            ]
            if sum(book_labels_for_sentence) > 1:
                multi_label_indices.append(i)
        
        logger.info(f"Found {len(multi_label_indices)} multi-label sentences")
        
        # For positive samples, prioritize multi-label sentences
        positive_multi_label = [i for i in positive_indices if i in multi_label_indices]
        positive_single_label = [i for i in positive_indices if i not in multi_label_indices]
        
        logger.info(f"Positive multi-label: {len(positive_multi_label)}, Positive single-label: {len(positive_single_label)}")
        
        # Include ALL multi-label sentences that are positive for this book
        selected_positive = positive_multi_label.copy()
        
        # Add single-label positive samples to reach target
        remaining_positive_needed = target_samples_per_class - len(selected_positive)
        if remaining_positive_needed > 0 and len(positive_single_label) > 0:
            # Sample from single-label positives
            n_to_sample = min(remaining_positive_needed, len(positive_single_label))
            selected_single_label = np.random.choice(positive_single_label, n_to_sample, replace=False)
            selected_positive.extend(selected_single_label)
        
        # For negative samples, we can sample normally since we want to balance the dataset
        n_negative = min(len(negative_indices), len(selected_positive))
        selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
        
        # Combine indices
        original_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(original_indices)
        
        # Create balanced dataset
        X_balanced = self.embeddings[original_indices]
        y_balanced = self.book_labels[book_col][original_indices]
        
        # Count multi-label sentences in the final dataset
        multi_label_in_dataset = sum(1 for i in original_indices if i in multi_label_indices)
        
        logger.info(f"Created balanced dataset with {len(X_balanced)} samples")
        logger.info(f"Positive samples: {np.sum(y_balanced == 1)}, Negative samples: {np.sum(y_balanced == 0)}")
        logger.info(f"Multi-label sentences in dataset: {multi_label_in_dataset}")
        
        return X_balanced, y_balanced, original_indices
    
    def train_book_model(self, book_col, hidden_dims=[256, 128, 64], dropout_rate=0.3,
                        epochs=100, batch_size=32, learning_rate=0.001, patience=15):
        """Train a model for a specific book."""
        book_name = self.book_names[book_col]
        logger.info(f"Training model for {book_name}")
        
        # Create balanced dataset
        X_balanced, y_balanced, original_indices = self.create_balanced_dataset(book_col)
        
        # Split data using the original indices
        train_size = int(0.7 * len(original_indices))
        val_size = int(0.15 * len(original_indices))
        
        train_indices = original_indices[:train_size]
        val_indices = original_indices[train_size:train_size + val_size]
        test_indices = original_indices[train_size + val_size:]
        
        # Create the actual data splits
        X_train = X_balanced[:train_size]
        y_train = y_balanced[:train_size]
        X_val = X_balanced[train_size:train_size + val_size]
        y_val = y_balanced[train_size:train_size + val_size]
        X_test = X_balanced[train_size + val_size:]
        y_test = y_balanced[train_size + val_size:]
        

        
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
        multi_label_results = self.evaluate_multi_label_performance(
            book_col, model, scaler, X_test_scaled, y_test, test_indices
        )
        
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
            'multi_label_results': multi_label_results
        }
        
        logger.info(f"{book_name} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return model, scaler, self.results[book_col]
    
    def train_all_models(self):
        """Train models for all books."""
        logger.info("Training models for all books...")
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            try:
                self.train_book_model(book_col)
            except Exception as e:
                logger.error(f"Error training model for {book_col}: {e}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training results."""
        results_path = "models/individual_book_results.json"
        
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
        plt.savefig('models/individual_book_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comparison plot saved to models/individual_book_model_comparison.png")

    def evaluate_multi_label_performance(self, book_col, model, scaler, X_test, y_test, test_indices):
        """Evaluate how well the model performs on multi-label sentences."""
        book_name = self.book_names[book_col]
        
        # Identify multi-label sentences in test set
        multi_label_test_indices = []
        for i, test_idx in enumerate(test_indices):
            # Count how many books this sentence belongs to
            book_labels_for_sentence = [
                self.book_labels['book_1'][test_idx],
                self.book_labels['book_2'][test_idx], 
                self.book_labels['book_3'][test_idx],
                self.book_labels['book_4'][test_idx]
            ]
            if sum(book_labels_for_sentence) > 1:
                multi_label_test_indices.append(i)
        
        if len(multi_label_test_indices) == 0:
            logger.info(f"No multi-label sentences in test set for {book_name}")
            return None
        
        # Get predictions for multi-label sentences
        model.eval()
        multi_label_predictions = []
        multi_label_targets = []
        
        with torch.no_grad():
            for i in multi_label_test_indices:
                X_sample = torch.FloatTensor(X_test[i:i+1]).to(self.device)
                output = model(X_sample)
                if output.dim() == 0:  # Handle scalar output
                    multi_label_predictions.append(output.cpu().numpy())
                else:
                    multi_label_predictions.append(output.cpu().numpy()[0])
                multi_label_targets.append(y_test[i])
        
        # Convert to numpy arrays
        multi_label_predictions = np.array(multi_label_predictions)
        multi_label_targets = np.array(multi_label_targets)
        
        # Convert probabilities to binary predictions
        multi_label_pred_binary = (multi_label_predictions > 0.5).astype(int)
        
        # Calculate metrics for multi-label sentences
        multi_label_accuracy = accuracy_score(multi_label_targets, multi_label_pred_binary)
        multi_label_precision = precision_score(multi_label_targets, multi_label_pred_binary)
        multi_label_recall = recall_score(multi_label_targets, multi_label_pred_binary)
        multi_label_f1 = f1_score(multi_label_targets, multi_label_pred_binary)
        
        logger.info(f"{book_name} Multi-label Performance:")
        logger.info(f"Multi-label sentences in test: {len(multi_label_test_indices)}")
        logger.info(f"Multi-label Accuracy: {multi_label_accuracy:.4f}")
        logger.info(f"Multi-label Precision: {multi_label_precision:.4f}")
        logger.info(f"Multi-label Recall: {multi_label_recall:.4f}")
        logger.info(f"Multi-label F1 Score: {multi_label_f1:.4f}")
        
        return {
            'multi_label_count': len(multi_label_test_indices),
            'multi_label_accuracy': multi_label_accuracy,
            'multi_label_precision': multi_label_precision,
            'multi_label_recall': multi_label_recall,
            'multi_label_f1': multi_label_f1,
            'multi_label_predictions': multi_label_predictions,
            'multi_label_targets': multi_label_targets
        }

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
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 