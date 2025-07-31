#!/usr/bin/env python3
"""
Train contrastive learning models for each book category (orchestration approach).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import yaml
import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.semantic_embedding_model import SemanticEmbeddingModel, ContrastiveLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookSpecificContrastiveModel(nn.Module):
    """Contrastive learning model for a specific book category."""
    
    def __init__(self, base_model: SemanticEmbeddingModel, book_name: str):
        super().__init__()
        self.base_model = base_model
        self.book_name = book_name
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add a projection head for this specific book
        self.projection_head = nn.Sequential(
            nn.Linear(self.base_model.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss()
        
    def forward(self, sentences: List[str]) -> torch.Tensor:
        """Forward pass to get embeddings."""
        embeddings = self.base_model.encode_sentences(sentences)
        # Move embeddings to the same device as projection head
        device = next(self.projection_head.parameters()).device
        embeddings = embeddings.to(device)
        projected = self.projection_head(embeddings)
        return projected
    
    def compute_loss(self, anchor_sentences: List[str], positive_sentences: List[str], 
                    negative_sentences: List[str]) -> torch.Tensor:
        """Compute contrastive loss for this book."""
        anchor_embeddings = self.forward(anchor_sentences)
        positive_embeddings = self.forward(positive_sentences)
        negative_embeddings = self.forward(negative_sentences)
        
        # Create labels (1 for positive pairs, 0 for negative pairs)
        labels = torch.ones(anchor_embeddings.size(0), device=anchor_embeddings.device)
        
        # Compute loss
        loss = self.contrastive_loss(anchor_embeddings, positive_embeddings, labels)
        loss += self.contrastive_loss(anchor_embeddings, negative_embeddings, 1 - labels)
        
        return loss

def load_semantic_model(model_path: str, config: dict, device: str = "cpu") -> SemanticEmbeddingModel:
    """Load the trained semantic embedding model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SemanticEmbeddingModel(
        model_name=checkpoint['model_name'],
        embedding_dim=checkpoint['embedding_dim']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def load_knn_features(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, Dict]:
    """Load KNN features and metadata."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load KNN features
    features_path = "data/features_knn/augmented_dataset.csv"
    if not Path(features_path).exists():
        raise FileNotFoundError(f"KNN features not found at {features_path}. Please run Step 4 first.")
    
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} samples with KNN features")
    
    # Load feature summary
    summary_path = "data/features_knn/feature_summary_improved.json"
    with open(summary_path, 'r') as f:
        feature_summary = json.load(f)
    
    return df, feature_summary

def prepare_book_specific_data(df: pd.DataFrame, book_name: str) -> Tuple[List[str], List[int]]:
    """Prepare data for a specific book category."""
    
    sentences = df['sentence'].tolist()
    belongs_col = f'belongs_to_{book_name}'
    
    # Create binary labels (1 if sentence belongs to this book, 0 otherwise)
    labels = (df[belongs_col] == 1).astype(int).tolist()
    
    return sentences, labels

def create_triplets_for_book(sentences: List[str], labels: List[int], 
                           book_name: str, num_triplets: int = 1000) -> List[Dict]:
    """Create triplets for contrastive learning for a specific book."""
    
    # Separate positive and negative samples
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    if len(positive_indices) < 2 or len(negative_indices) < 1:
        logger.warning(f"Not enough samples for book {book_name}")
        return []
    
    triplets = []
    
    for _ in range(num_triplets):
        # Sample anchor and positive from positive samples
        anchor_idx = np.random.choice(positive_indices)
        positive_idx = np.random.choice(positive_indices)
        
        # Sample negative from negative samples
        negative_idx = np.random.choice(negative_indices)
        
        # Ensure anchor and positive are different
        while positive_idx == anchor_idx:
            positive_idx = np.random.choice(positive_indices)
        
        triplet = {
            'anchor': sentences[anchor_idx],
            'positive': sentences[positive_idx],
            'negative': sentences[negative_idx],
            'book': book_name
        }
        triplets.append(triplet)
    
    return triplets

def train_book_model(model: BookSpecificContrastiveModel, triplets: List[Dict],
                    val_sentences: List[str], val_labels: List[int],
                    num_epochs: int = 20, learning_rate: float = 1e-4,
                    device: str = "cpu") -> Dict:
    """Train a book-specific contrastive model."""
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    logger.info(f"Training model for {model.book_name} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        # Sample a batch of triplets
        batch_size = min(32, len(triplets))
        batch_triplets = np.random.choice(triplets, batch_size, replace=False)
        
        anchor_sentences = [t['anchor'] for t in batch_triplets]
        positive_sentences = [t['positive'] for t in batch_triplets]
        negative_sentences = [t['negative'] for t in batch_triplets]
        
        optimizer.zero_grad()
        loss = model.compute_loss(anchor_sentences, positive_sentences, negative_sentences)
        loss.backward()
        optimizer.step()
        
        epoch_loss = loss.item()
        train_losses.append(epoch_loss)
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_accuracy = evaluate_book_model(model, val_sentences, val_labels, device)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'final_train_loss': train_losses[-1],
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0
    }

def evaluate_book_model(model: BookSpecificContrastiveModel, test_sentences: List[str],
                      test_labels: List[int], device: str = "cpu") -> float:
    """Evaluate a book-specific model."""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Get embeddings for test sentences
        embeddings = model.forward(test_sentences)
        
        # Simple threshold-based classification
        # We'll use the magnitude of embeddings as a proxy for belonging
        embedding_magnitudes = torch.norm(embeddings, dim=1)
        threshold = torch.median(embedding_magnitudes)
        
        predictions = (embedding_magnitudes > threshold).float().cpu().numpy()
    
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def train_contrastive_orchestration(config_path: str = "configs/config.yaml",
                                  output_dir: str = "experiments/contrastive_orchestration",
                                  device: str = "auto") -> Dict:
    """Train contrastive learning models for each book category."""
    
    # Set device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load semantic model
    semantic_model_path = "experiments/semantic_embedding/semantic_embedding_model.pt"
    if not Path(semantic_model_path).exists():
        raise FileNotFoundError(f"Semantic model not found at {semantic_model_path}")
    
    semantic_model = load_semantic_model(semantic_model_path, config, device)
    logger.info("Semantic model loaded successfully!")
    
    # Load KNN features
    df, feature_summary = load_knn_features(config_path)
    books = feature_summary['books']
    
    # Train a model for each book
    models = {}
    training_results = {}
    
    for book_name in books:
        logger.info(f"\n=== Training model for {book_name} ===")
        
        # Prepare data for this book
        sentences, labels = prepare_book_specific_data(df, book_name)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            sentences, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Positive samples in train: {sum(y_train)}/{len(y_train)}")
        
        # Create triplets for training
        train_triplets = create_triplets_for_book(X_train, y_train, book_name, num_triplets=500)
        logger.info(f"Created {len(train_triplets)} triplets for training")
        
        # Create model for this book
        book_model = BookSpecificContrastiveModel(semantic_model, book_name)
        
        # Train the model
        results = train_book_model(
            model=book_model,
            triplets=train_triplets,
            val_sentences=X_val,
            val_labels=y_val,
            num_epochs=20,
            learning_rate=1e-4,
            device=device
        )
        
        # Evaluate on test set
        test_accuracy = evaluate_book_model(book_model, X_test, y_test, device)
        results['test_accuracy'] = test_accuracy
        
        models[book_name] = book_model
        training_results[book_name] = results
        
        logger.info(f"Test accuracy for {book_name}: {test_accuracy:.4f}")
    
    # Save models and results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each model separately
    for book_name, model in models.items():
        torch.save({
            'model_state_dict': model.state_dict(),
            'book_name': book_name,
            'training_results': training_results[book_name],
            'base_model_path': semantic_model_path
        }, output_path / f"contrastive_model_{book_name.replace(' ', '_')}.pt")
    
    # Save overall results
    overall_results = {
        'books': books,
        'training_results': training_results,
        'average_test_accuracy': np.mean([results['test_accuracy'] for results in training_results.values()])
    }
    
    with open(output_path / "overall_results.json", 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Print summary
    print(f"\n=== CONTRASTIVE ORCHESTRATION TRAINING COMPLETED ===")
    print(f"Models saved to: {output_path}")
    
    print(f"\n=== RESULTS SUMMARY ===")
    for book_name, results in training_results.items():
        print(f"{book_name}:")
        print(f"  Final Train Loss: {results['final_train_loss']:.4f}")
        print(f"  Final Val Accuracy: {results['final_val_accuracy']:.4f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    
    print(f"\nAverage Test Accuracy: {overall_results['average_test_accuracy']:.4f}")
    
    return overall_results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train contrastive learning models for each book")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", default="experiments/contrastive_orchestration", 
                       help="Output directory for models and logs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    try:
        results = train_contrastive_orchestration(
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device
        )
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 