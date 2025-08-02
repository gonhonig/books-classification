"""
Optimized Neural Network for Multi-Label Classification
Focuses on improving multi-label performance with advanced techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
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

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant semantic features."""
    
    def __init__(self, input_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.output_proj(attended.squeeze(1))

class OptimizedMultiLabelMLP(nn.Module):
    """Optimized MLP with attention and residual connections."""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], output_dim=4, dropout_rate=0.3):
        super(OptimizedMultiLabelMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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
        
        # Attention layer
        self.attention = AttentionLayer(prev_dim)
        
        # Multi-head output layers (separate for each book)
        self.book_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(output_dim)
        ])
        
    def forward(self, x):
        # Feature extraction
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        # Apply attention
        attended_features = self.attention(features.unsqueeze(1)).squeeze(1)
        
        # Multi-head output
        outputs = []
        for head in self.book_heads:
            outputs.append(head(attended_features))
        
        return torch.cat(outputs, dim=1)

class OptimizedNeuralNetworkTrainer:
    """Optimized trainer with multi-label specific improvements."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load and prepare the dataset with class-balanced sampling."""
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
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        self.X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Create class-balanced sampler
        self.create_balanced_sampler()
        
        logger.info("Data preparation completed!")
        return True
    
    def create_balanced_sampler(self):
        """Create a balanced sampler to handle class imbalance."""
        # Calculate sample weights based on label count
        label_counts = torch.sum(self.y_train, dim=1)
        
        # Give higher weight to multi-label samples
        weights = torch.ones(len(self.y_train))
        weights[label_counts > 1] = 3.0  # 3x weight for multi-label samples
        weights[label_counts > 2] = 5.0  # 5x weight for 3+ label samples
        
        self.sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    
    def create_model(self, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """Create the optimized neural network model."""
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        
        self.model = OptimizedMultiLabelMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Created optimized model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_model(self, epochs=150, batch_size=32, learning_rate=0.001, patience=20):
        """Train the optimized neural network."""
        if self.model is None:
            self.create_model()
        
        # Use Focal Loss for better multi-label performance
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
        
        # Create data loader with balanced sampling
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=self.sampler)
        
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
        logger.info("Using Focal Loss and class-balanced sampling")
        
        best_val_loss = float('inf')
        best_multi_label_f1 = 0.0
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
                
                # Multi-label specific metrics
                multi_label_mask = np.sum(val_true, axis=1) > 1
                if np.sum(multi_label_mask) > 0:
                    multi_label_f1 = f1_score(val_true[multi_label_mask], val_pred[multi_label_mask], 
                                            average='weighted', zero_division=0)
                else:
                    multi_label_f1 = 0.0
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping based on multi-label F1 score
            if multi_label_f1 > best_multi_label_f1:
                best_multi_label_f1 = multi_label_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_optimized_neural_network.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                          f"Val F1: {val_f1:.4f}, Multi-label F1: {multi_label_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_optimized_neural_network.pth'))
        logger.info("Training completed!")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_multi_label_f1': best_multi_label_f1
        }
    
    def evaluate_model(self):
        """Evaluate the trained model with focus on multi-label performance."""
        if self.model is None:
            logger.error("No model to evaluate!")
            return
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            predictions = (outputs > 0.5).float().cpu().numpy()
            probabilities = outputs.cpu().numpy()
            true_labels = self.y_test.cpu().numpy()
        
        # Calculate overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        hamming = hamming_loss(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Multi-label specific metrics
        multi_label_mask = np.sum(true_labels, axis=1) > 1
        multi_label_accuracy = accuracy_score(true_labels[multi_label_mask], predictions[multi_label_mask]) if np.sum(multi_label_mask) > 0 else 0
        multi_label_f1 = f1_score(true_labels[multi_label_mask], predictions[multi_label_mask], average='weighted', zero_division=0) if np.sum(multi_label_mask) > 0 else 0
        
        # Average predictions per sample
        avg_predictions = np.mean(np.sum(predictions, axis=1))
        
        # Analyze performance by label count
        test_label_counts = np.sum(true_labels, axis=1)
        
        logger.info("Optimized Neural Network Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        logger.info(f"  Overall F1 Score: {f1:.4f}")
        logger.info(f"  Multi-label Accuracy: {multi_label_accuracy:.4f}")
        logger.info(f"  Multi-label F1 Score: {multi_label_f1:.4f}")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
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
            'f1_score': f1,
            'multi_label_accuracy': multi_label_accuracy,
            'multi_label_f1': multi_label_f1,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
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
        model_path = models_dir / "optimized_neural_network_semantic.pth"
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
        results_path = models_dir / "optimized_neural_network_results.json"
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
        
        logger.info(f"Optimized model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")

def main():
    """Main training function."""
    logger.info("Starting optimized neural network training...")
    
    # Initialize trainer
    trainer = OptimizedNeuralNetworkTrainer()
    
    # Load data
    if not trainer.load_data():
        logger.error("Failed to load data!")
        return
    
    # Train model
    training_history = trainer.train_model(
        epochs=150,
        batch_size=32,
        learning_rate=0.001,
        patience=20
    )
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    logger.info(f"Training completed!")
    logger.info(f"Overall F1 Score: {results['f1_score']:.4f}")
    logger.info(f"Multi-label F1 Score: {results['multi_label_f1']:.4f}")

if __name__ == "__main__":
    main() 