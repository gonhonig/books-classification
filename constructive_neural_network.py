"""
Constructive Learning Neural Network for Multi-Label Classification
Uses progressive learning and advanced techniques to achieve 75%+ accuracy on all multi-label variations.
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
import copy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    """Transformer block for better semantic understanding."""
    
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification."""
    
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.clip > 0:
            loss = loss * (1 - self.clip)

        return -loss.mean()

class ConstructiveMultiLabelMLP(nn.Module):
    """Advanced MLP with transformer blocks and constructive learning."""
    
    def __init__(self, input_dim=384, hidden_dims=[512, 256, 128], output_dim=4, dropout_rate=0.2):
        super(ConstructiveMultiLabelMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature extraction with transformer blocks
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            self.feature_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ))
            
            # Transformer block for semantic understanding
            if hidden_dim >= 128:  # Only add transformer for larger dimensions
                self.feature_layers.append(TransformerBlock(hidden_dim, n_heads=8, d_ff=hidden_dim*2, dropout=dropout_rate))
            
            prev_dim = hidden_dim
        
        # Global attention mechanism
        self.global_attention = nn.MultiheadAttention(prev_dim, num_heads=8, batch_first=True)
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(prev_dim * 2, prev_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_dim, prev_dim // 2)
        )
        
        # Separate specialized heads for each book
        self.book_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim // 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(output_dim)
        ])
        
        # Label correlation layer
        self.label_correlation = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction with transformer blocks
        features = x
        transformer_features = []
        
        for i, layer in enumerate(self.feature_layers):
            if isinstance(layer, TransformerBlock):
                # Reshape for transformer
                batch_size = features.size(0)
                features = features.unsqueeze(1)  # Add sequence dimension
                features = layer(features)
                features = features.squeeze(1)  # Remove sequence dimension
                transformer_features.append(features)
            else:
                features = layer(features)
                transformer_features.append(features)
        
        # Global attention
        features_attended = features.unsqueeze(1)
        attended_features, _ = self.global_attention(features_attended, features_attended, features_attended)
        attended_features = attended_features.squeeze(1)
        
        # Feature fusion
        fused_features = torch.cat([features, attended_features], dim=1)
        final_features = self.feature_fusion(fused_features)
        
        # Multi-head output
        outputs = []
        for head in self.book_heads:
            outputs.append(head(final_features))
        
        # Combine outputs
        combined_outputs = torch.cat(outputs, dim=1)
        
        # Apply label correlation
        correlated_outputs = self.label_correlation(combined_outputs)
        
        return correlated_outputs

class ConstructiveNeuralNetworkTrainer:
    """Constructive learning trainer with progressive complexity."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        self.data_path = data_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load and prepare the dataset with constructive learning setup."""
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
        
        # Analyze label distribution for constructive learning
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
        
        # Create constructive learning datasets
        self.create_constructive_datasets()
        
        logger.info("Data preparation completed!")
        return True
    
    def create_constructive_datasets(self):
        """Create datasets for constructive learning stages."""
        train_label_counts = torch.sum(self.y_train, dim=1)
        
        # Stage 1: Single-label samples
        single_label_mask = train_label_counts == 1
        self.single_label_X = self.X_train[single_label_mask]
        self.single_label_y = self.y_train[single_label_mask]
        
        # Stage 2: 2-label samples
        two_label_mask = train_label_counts == 2
        self.two_label_X = self.X_train[two_label_mask]
        self.two_label_y = self.y_train[two_label_mask]
        
        # Stage 3: 3+ label samples
        multi_label_mask = train_label_counts >= 3
        self.multi_label_X = self.X_train[multi_label_mask]
        self.multi_label_y = self.y_train[multi_label_mask]
        
        logger.info(f"Constructive learning datasets:")
        logger.info(f"  Stage 1 (Single-label): {len(self.single_label_X)} samples")
        logger.info(f"  Stage 2 (2-label): {len(self.two_label_X)} samples")
        logger.info(f"  Stage 3 (3+ label): {len(self.multi_label_X)} samples")
    
    def create_model(self, hidden_dims=[512, 256, 128], dropout_rate=0.2):
        """Create the constructive neural network model."""
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        
        self.model = ConstructiveMultiLabelMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Created constructive model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_stage(self, stage_name, X, y, epochs=50, batch_size=32, learning_rate=0.001):
        """Train a specific stage of constructive learning."""
        logger.info(f"Training Stage: {stage_name}")
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_loss = train_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}: Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        logger.info(f"  Stage {stage_name} completed with best loss: {best_loss:.4f}")
    
    def train_constructive_model(self, epochs_per_stage=50):
        """Train using constructive learning approach."""
        if self.model is None:
            self.create_model()
        
        logger.info("Starting constructive learning training...")
        
        # Stage 1: Train on single-label samples
        self.train_stage("Single-label", self.single_label_X, self.single_label_y, 
                        epochs=epochs_per_stage, learning_rate=0.001)
        
        # Stage 2: Continue training on 2-label samples
        combined_X_2 = torch.cat([self.single_label_X, self.two_label_X], dim=0)
        combined_y_2 = torch.cat([self.single_label_y, self.two_label_y], dim=0)
        self.train_stage("2-label", combined_X_2, combined_y_2, 
                        epochs=epochs_per_stage, learning_rate=0.0005)
        
        # Stage 3: Final training on all samples
        all_X = torch.cat([self.single_label_X, self.two_label_X, self.multi_label_X], dim=0)
        all_y = torch.cat([self.single_label_y, self.two_label_y, self.multi_label_y], dim=0)
        self.train_stage("All samples", all_X, all_y, 
                        epochs=epochs_per_stage, learning_rate=0.0001)
        
        logger.info("Constructive learning training completed!")
    
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
        test_label_counts = np.sum(true_labels, axis=1)
        
        logger.info("Constructive Neural Network Results:")
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
        model_path = models_dir / "constructive_neural_network.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'results': self.results,
            'model_config': {
                'input_dim': self.X_train.shape[1],
                'output_dim': self.y_train.shape[1],
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.2
            }
        }, model_path)
        
        # Save results
        results_path = models_dir / "constructive_neural_network_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays and float32 to JSON serializable types
            results_for_json = {}
            for k, v in self.results.items():
                if isinstance(v, np.ndarray):
                    results_for_json[k] = v.tolist()
                elif isinstance(v, np.float32):
                    results_for_json[k] = float(v)
                elif isinstance(v, dict):
                    results_for_json[k] = {str(kk): float(vv) for kk, vv in v.items()}
                else:
                    results_for_json[k] = v
            json.dump(results_for_json, f, indent=2)
        
        logger.info(f"Constructive model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")

def main():
    """Main training function."""
    logger.info("Starting constructive neural network training...")
    
    # Initialize trainer
    trainer = ConstructiveNeuralNetworkTrainer()
    
    # Load data
    if not trainer.load_data():
        logger.error("Failed to load data!")
        return
    
    # Train model using constructive learning
    trainer.train_constructive_model(epochs_per_stage=50)
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    logger.info(f"Training completed!")
    logger.info(f"Overall F1 Score: {results['f1_score']:.4f}")
    logger.info(f"Target achieved: {results['target_achieved']}")

if __name__ == "__main__":
    main() 