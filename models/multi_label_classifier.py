#!/usr/bin/env python3
"""
Multi-label classification model for book sentence classification.
Uses semantic embeddings to predict which books a sentence could belong to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MultiLabelClassifier(nn.Module):
    """Multi-label classifier that predicts book belonging scores."""
    
    def __init__(self, embedding_dim: int = 384, num_books: int = 4, 
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_books = num_books
        self.hidden_size = hidden_size
        
        # Classification head
        layers = []
        input_dim = embedding_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        
        # Final output layer - one score per book
        layers.append(nn.Linear(hidden_size, num_books))
        layers.append(nn.Sigmoid())  # Output probabilities between 0 and 1
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-label classification.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            book_scores: Tensor of shape (batch_size, num_books) with belonging scores
        """
        return self.classifier(embeddings)

class SemanticMultiLabelModel(nn.Module):
    """Complete model combining semantic embedding and multi-label classification."""
    
    def __init__(self, semantic_model: nn.Module, num_books: int = 4,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.semantic_model = semantic_model
        self.classifier = MultiLabelClassifier(
            embedding_dim=semantic_model.embedding_dim,
            num_books=num_books,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, sentences: List[str], return_embeddings: bool = False) -> Dict:
        """
        Forward pass through semantic embedding and classification.
        
        Args:
            sentences: List of sentences to classify
            return_embeddings: Whether to return embeddings as well
            
        Returns:
            Dictionary with 'book_scores' and optionally 'embeddings'
        """
        # Get semantic embeddings
        with torch.no_grad():
            embeddings = self.semantic_model.encode_sentences(sentences)
        
        # Move embeddings to the same device as classifier
        device = next(self.classifier.parameters()).device
        embeddings = embeddings.to(device)
        
        # Get book belonging scores
        book_scores = self.classifier(embeddings)
        
        result = {'book_scores': book_scores}
        
        if return_embeddings:
            result['embeddings'] = embeddings
            
        return result

class MultiLabelDataset:
    """Dataset for multi-label classification training."""
    
    def __init__(self, sentences: List[str], book_labels: List[str], 
                 book_to_id: Dict[str, int], semantic_model: nn.Module):
        self.sentences = sentences
        self.book_labels = book_labels
        self.book_to_id = book_to_id
        self.semantic_model = semantic_model
        
        # Create multi-label targets
        self.targets = self._create_multi_label_targets()
        
    def _create_multi_label_targets(self) -> torch.Tensor:
        """Create multi-label targets based on original book and semantic similarity."""
        num_books = len(self.book_to_id)
        targets = torch.zeros(len(self.sentences), num_books)
        
        # Get embeddings for similarity calculation
        with torch.no_grad():
            embeddings = self.semantic_model.encode_sentences(self.sentences)
        
        # Calculate similarity to each book's sentences
        for i, sentence in enumerate(self.sentences):
            original_book = self.book_labels[i]
            original_book_id = self.book_to_id[original_book]
            
            # Set high score for original book
            targets[i, original_book_id] = 1.0
            
            # Calculate similarity to other books
            for book_name, book_id in self.book_to_id.items():
                if book_name != original_book:
                    # Find sentences from this book
                    book_sentences = [j for j, label in enumerate(self.book_labels) 
                                    if label == book_name]
                    
                    if book_sentences:
                        # Calculate average similarity to this book's sentences
                        book_embeddings = embeddings[book_sentences]
                        current_embedding = embeddings[i:i+1]
                        
                        similarities = torch.cosine_similarity(
                            current_embedding, book_embeddings, dim=1
                        )
                        avg_similarity = torch.mean(similarities).item()
                        
                        # Set score based on similarity (threshold at 0.6)
                        if avg_similarity > 0.6:
                            targets[i, book_id] = avg_similarity
        
        return targets
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.sentences[idx], self.targets[idx]

def create_multi_label_dataloader(sentences: List[str], book_labels: List[str],
                                book_to_id: Dict[str, int], semantic_model: nn.Module,
                                batch_size: int = 32, shuffle: bool = True):
    """Create dataloader for multi-label classification."""
    from torch.utils.data import DataLoader
    
    dataset = MultiLabelDataset(sentences, book_labels, book_to_id, semantic_model)
    
    def collate_fn(batch):
        sentences, targets = zip(*batch)
        return list(sentences), torch.stack(targets)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def train_multi_label_classifier(semantic_model: nn.Module, 
                               sentences: List[str], 
                               book_labels: List[str],
                               book_to_id: Dict[str, int],
                               config: Dict,
                               device: str = "auto") -> nn.Module:
    """
    Train multi-label classifier on semantic embeddings.
    
    Args:
        semantic_model: Pre-trained semantic embedding model
        sentences: Training sentences
        book_labels: Original book labels
        book_to_id: Mapping from book names to IDs
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained multi-label classifier
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    logger.info(f"Training multi-label classifier on device: {device}")
    
    # Create complete model
    model = SemanticMultiLabelModel(
        semantic_model=semantic_model,
        num_books=len(book_to_id),
        hidden_size=config['model']['semantic_embedding']['multi_label_classifier']['hidden_size'],
        num_layers=config['model']['semantic_embedding']['multi_label_classifier']['num_layers'],
        dropout=config['model']['semantic_embedding']['multi_label_classifier']['dropout']
    )
    model.to(device)
    
    # Create dataloader
    dataloader = create_multi_label_dataloader(
        sentences=sentences,
        book_labels=book_labels,
        book_to_id=book_to_id,
        semantic_model=semantic_model,
        batch_size=config['training']['batch_size']
    )
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=float(config['model']['training_phases'][1]['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Training loop
    num_epochs = config['model']['training_phases'][1]['epochs']
    
    logger.info(f"Starting multi-label training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (sentences_batch, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(sentences_batch)
            book_scores = outputs['book_scores']
            
            # Calculate loss
            loss = criterion(book_scores, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    logger.info("Multi-label training completed!")
    return model

def evaluate_multi_label_model(model: nn.Module, test_sentences: List[str], 
                             test_labels: List[str], book_to_id: Dict[str, int],
                             threshold: float = 0.5) -> Dict:
    """Evaluate multi-label classification performance."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(test_sentences)
        book_scores = outputs['book_scores']
        
        # Convert to predictions using threshold
        predictions = (book_scores > threshold).float()
        
        # Calculate metrics
        total_correct = 0
        total_predictions = 0
        
        for i, sentence in enumerate(test_sentences):
            original_book = test_labels[i]
            original_book_id = book_to_id[original_book]
            
            # Check if original book is predicted
            if predictions[i, original_book_id] > 0:
                total_correct += 1
            
            # Count total predictions
            total_predictions += torch.sum(predictions[i]).item()
        
        accuracy = total_correct / len(test_sentences)
        avg_predictions = total_predictions / len(test_sentences)
        
        # Calculate per-book metrics
        book_metrics = {}
        for book_name, book_id in book_to_id.items():
            book_sentences = [i for i, label in enumerate(test_labels) if label == book_name]
            
            if book_sentences:
                book_correct = 0
                for idx in book_sentences:
                    if predictions[idx, book_id] > 0:
                        book_correct += 1
                
                book_accuracy = book_correct / len(book_sentences)
                book_metrics[book_name] = book_accuracy
        
        return {
            'overall_accuracy': accuracy,
            'avg_predictions_per_sentence': avg_predictions,
            'book_metrics': book_metrics,
            'predictions': predictions.cpu().numpy(),
            'scores': book_scores.cpu().numpy()
        }

if __name__ == "__main__":
    # Test the multi-label classifier
    import yaml
    
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load semantic model
    from models.semantic_embedding_model import SemanticEmbeddingModel
    
    semantic_model = SemanticEmbeddingModel(
        model_name=config['model']['encoder']['model_name'],
        embedding_dim=config['model']['encoder']['hidden_size']
    )
    
    # Test sentences
    test_sentences = [
        "What could I do?",
        "The monster approached slowly.",
        "Alice fell down the rabbit hole.",
        "Caesar crossed the Rubicon."
    ]
    
    # Create multi-label model
    model = SemanticMultiLabelModel(semantic_model, num_books=4)
    
    # Test forward pass
    outputs = model(test_sentences)
    book_scores = outputs['book_scores']
    
    print("Multi-label classification test:")
    for i, sentence in enumerate(test_sentences):
        scores = book_scores[i]
        print(f"'{sentence}': {scores}")
    
    print("Multi-label classifier test completed!") 