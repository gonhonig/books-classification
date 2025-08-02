"""
Ensemble Neural Network for Multi-Label Classification
Uses specialized models and threshold optimization to achieve 75%+ accuracy.
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

class DiceLoss(nn.Module):
    """Dice Loss for better multi-label performance."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class SpecializedMLP(nn.Module):
    """Specialized MLP for multi-label classification."""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], output_dim=4, dropout_rate=0.3):
        super(SpecializedMLP, self).__init__()
        
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
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class EnsembleNeuralNetworkTrainer:
    """Ensemble trainer with specialized models and threshold optimization."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.models = {}
        self.thresholds = {}
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
        
        # Analyze label distribution
        label_counts = np.sum(y, axis=1)
        logger.info(f"Label distribution:")
        logger.info(f"  Single-label samples: {np.sum(label_counts == 1)}")
        logger.info(f"  2-label samples: {np.sum(label_counts == 2)}")
        logger.info(f"  3-label samples: {np.sum(label_counts == 3)}")
        logger.info(f"  4-label samples: {np.sum(label_counts == 4)}")
        
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
        
        logger.info("Data preparation completed!")
        return True
    
    def create_model(self, model_name="main"):
        """Create a specialized neural network model."""
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        
        self.models[model_name] = SpecializedMLP(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            output_dim=output_dim,
            dropout_rate=0.3
        ).to(self.device)
        
        logger.info(f"Created {model_name} model with {sum(p.numel() for p in self.models[model_name].parameters())} parameters")
        return self.models[model_name]
    
    def train_model(self, model_name="main", epochs=100, batch_size=64, learning_rate=0.001):
        """Train a single model."""
        if model_name not in self.models:
            self.create_model(model_name)
        
        model = self.models[model_name]
        
        # Use Dice Loss for better multi-label performance
        criterion = DiceLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.7)
        
        # Create data loader
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Training {model_name} model for {epochs} epochs...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_loss = train_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}: Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
        logger.info(f"  {model_name} model training completed with best loss: {best_loss:.4f}")
    
    def optimize_thresholds(self, model_name="main"):
        """Optimize prediction thresholds for each book."""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            outputs = model(self.X_train)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            true_labels = self.y_train.cpu().numpy()
        
        # Optimize thresholds for each book
        thresholds = {}
        for book_idx in range(probabilities.shape[1]):
            best_threshold = 0.5
            best_f1 = 0.0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                predictions = (probabilities[:, book_idx] > threshold).astype(int)
                f1 = f1_score(true_labels[:, book_idx], predictions, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            thresholds[f'book_{book_idx}'] = best_threshold
            logger.info(f"  Book {book_idx}: optimal threshold = {best_threshold:.3f}, F1 = {best_f1:.4f}")
        
        self.thresholds[model_name] = thresholds
        return thresholds
    
    def train_ensemble(self):
        """Train multiple specialized models."""
        logger.info("Training ensemble of specialized models...")
        
        # Train main model
        self.train_model("main", epochs=100, learning_rate=0.001)
        
        # Train specialized models for different label counts
        train_label_counts = torch.sum(self.y_train, dim=1)
        
        # Model for 2+ labels
        multi_label_mask = train_label_counts >= 2
        if torch.sum(multi_label_mask) > 0:
            logger.info("Training specialized model for multi-label samples...")
            self.X_train_multi = self.X_train[multi_label_mask]
            self.y_train_multi = self.y_train[multi_label_mask]
            
            # Create and train multi-label model
            self.create_model("multi_label")
            self.train_model("multi_label", epochs=80, learning_rate=0.001)
        
        # Optimize thresholds for each model
        logger.info("Optimizing thresholds...")
        self.optimize_thresholds("main")
        if "multi_label" in self.models:
            self.optimize_thresholds("multi_label")
    
    def predict_ensemble(self, X):
        """Make ensemble predictions."""
        predictions = {}
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                # Apply optimized thresholds
                thresholds = self.thresholds[model_name]
                pred = np.zeros_like(probabilities)
                
                for book_idx in range(probabilities.shape[1]):
                    threshold = thresholds[f'book_{book_idx}']
                    pred[:, book_idx] = (probabilities[:, book_idx] > threshold).astype(int)
                
                predictions[model_name] = pred
        
        # Combine predictions (simple averaging for now)
        if len(predictions) > 1:
            combined_pred = np.mean(list(predictions.values()), axis=0)
            final_pred = (combined_pred > 0.5).astype(int)
        else:
            final_pred = list(predictions.values())[0]
        
        return final_pred
    
    def evaluate_model(self):
        """Evaluate the ensemble model."""
        if not self.models:
            logger.error("No models to evaluate!")
            return
        
        # Make ensemble predictions
        predictions = self.predict_ensemble(self.X_test)
        true_labels = self.y_test.cpu().numpy()
        
        # Calculate overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        hamming = hamming_loss(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Multi-label specific metrics
        test_label_counts = np.sum(true_labels, axis=1)
        
        logger.info("Ensemble Neural Network Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        logger.info(f"  Overall F1 Score: {f1:.4f}")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        
        logger.info("\nPerformance by label count:")
        multi_label_accuracies = {}
        for label_count in sorted(set(test_label_counts)):
            mask = test_label_counts == label_count
            if np.sum(mask) > 0:
                subset_accuracy = accuracy_score(true_labels[mask], predictions[mask])
                subset_f1 = f1_score(true_labels[mask], predictions[mask], average='weighted', zero_division=0)
                multi_label_accuracies[label_count] = subset_accuracy
                logger.info(f"  {label_count} label(s): {np.sum(mask)} samples, Accuracy: {subset_accuracy:.4f}, F1: {subset_f1:.4f}")
        
        # Check if we achieved 75% accuracy for all multi-label variations
        multi_label_achieved = all(acc >= 0.75 for label_count, acc in multi_label_accuracies.items() if label_count > 1)
        
        if multi_label_achieved:
            logger.info("🎉 SUCCESS: Achieved 75%+ accuracy for all multi-label variations!")
        else:
            logger.info("⚠️  Still working towards 75% accuracy for all multi-label variations")
        
        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
            'multi_label_accuracies': multi_label_accuracies,
            'target_achieved': multi_label_achieved,
            'predictions': predictions,
            'thresholds': self.thresholds
        }
        
        return self.results
    
    def save_model(self):
        """Save the ensemble model and results."""
        if not self.models:
            logger.error("No models to save!")
            return
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save all models
        for model_name, model in self.models.items():
            model_path = models_dir / f"ensemble_{model_name}_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scaler,
                'thresholds': self.thresholds.get(model_name, {}),
                'model_config': {
                    'input_dim': self.X_train.shape[1],
                    'output_dim': self.y_train.shape[1],
                    'hidden_dims': [256, 128, 64],
                    'dropout_rate': 0.3
                }
            }, model_path)
            logger.info(f"Saved {model_name} model to: {model_path}")
        
        # Save results
        results_path = models_dir / "ensemble_neural_network_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays and float32 to JSON serializable types
            results_for_json = {}
            for k, v in self.results.items():
                if isinstance(v, np.ndarray):
                    results_for_json[k] = v.tolist()
                elif isinstance(v, np.float32):
                    results_for_json[k] = float(v)
                elif isinstance(v, dict):
                    if k == 'thresholds':
                        # Handle nested thresholds dictionary
                        results_for_json[k] = {}
                        for model_name, model_thresholds in v.items():
                            results_for_json[k][model_name] = {str(kk): float(vv) for kk, vv in model_thresholds.items()}
                    else:
                        results_for_json[k] = {str(kk): float(vv) for kk, vv in v.items()}
                else:
                    results_for_json[k] = v
            json.dump(results_for_json, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")

def main():
    """Main training function."""
    logger.info("Starting ensemble neural network training...")
    
    # Initialize trainer
    trainer = EnsembleNeuralNetworkTrainer()
    
    # Load data
    if not trainer.load_data():
        logger.error("Failed to load data!")
        return
    
    # Train ensemble
    trainer.train_ensemble()
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    logger.info(f"Training completed!")
    logger.info(f"Overall F1 Score: {results['f1_score']:.4f}")
    logger.info(f"Target achieved: {results['target_achieved']}")

if __name__ == "__main__":
    main() 