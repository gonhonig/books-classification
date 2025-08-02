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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import json
from datetime import datetime

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

class NeuralNetworkTrainer:
    """Trainer for neural network multi-label classification."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load dataset
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Load semantic embeddings
        logger.info("Loading semantic embeddings...")
        embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')['embeddings']
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Verify embeddings match
        if len(df) != len(embeddings):
            logger.error(f"ERROR: Shape mismatch! Dataset: {len(df)}, Embeddings: {len(embeddings)}")
            return False
        
        logger.info("✓ Embeddings verified!")
        
        # Prepare features and labels
        X = embeddings
        label_cols = [col for col in df.columns if col.startswith('book_')]
        y = df[label_cols].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Check label distribution
        unique_labels_per_sample = np.sum(y, axis=1)
        logger.info(f"Label distribution:")
        logger.info(f"  Single-label samples: {np.sum(unique_labels_per_sample == 1)}")
        logger.info(f"  Multi-label samples: {np.sum(unique_labels_per_sample > 1)}")
        logger.info(f"  Average labels per sample: {np.mean(unique_labels_per_sample):.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        self.X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        logger.info("Data preparation completed!")
        return True
    
    def create_model(self, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """Create the neural network model."""
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        
        self.model = MultiLabelMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_model(self, epochs=100, batch_size=64, learning_rate=0.001, patience=10):
        """Train the neural network."""
        if self.model is None:
            self.create_model()
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Create data loader
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
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
            with torch.no_grad():
                val_outputs = self.model(self.X_test)
                val_loss = criterion(val_outputs, self.y_test).item()
                
                # Convert to numpy for metrics
                val_pred = (val_outputs > 0.5).float().cpu().numpy()
                val_true = self.y_test.cpu().numpy()
                
                val_accuracy = accuracy_score(val_true, val_pred)
                val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_neural_network.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_neural_network.pth'))
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
    
    def save_model(self):
        """Save the trained model and results."""
        if self.model is None:
            logger.error("No model to save!")
            return
        
        # Create models directory
        models_dir = Path("models")
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
    
    # Save model
    trainer.save_model()
    
    logger.info(f"Training completed! Final F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main() 