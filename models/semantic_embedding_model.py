"""
Semantic Embedding Model for Contrastive Learning
Trains sentence embeddings using cross-book similar pairs for semantic similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for training semantic embeddings."""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: First sentence embeddings [batch_size, embedding_dim]
            embeddings2: Second sentence embeddings [batch_size, embedding_dim]
            labels: Similarity labels [batch_size] (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarities between pairs
        similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1) / self.temperature
        
        # Positive pairs loss (bring similar sentences closer)
        positive_mask = labels == 1
        if positive_mask.sum() > 0:
            positive_similarities = similarities[positive_mask]
            positive_loss = -torch.log(torch.exp(positive_similarities) + 1e-8).mean()
        else:
            positive_loss = torch.tensor(0.0, device=embeddings1.device)
        
        # Negative pairs loss (push dissimilar sentences apart)
        negative_mask = labels == 0
        if negative_mask.sum() > 0:
            negative_similarities = similarities[negative_mask]
            negative_loss = torch.log(1 + torch.exp(negative_similarities - self.margin)).mean()
        else:
            negative_loss = torch.tensor(0.0, device=embeddings1.device)
        
        return positive_loss + negative_loss

class SemanticEmbeddingModel(nn.Module):
    """Semantic embedding model using contrastive learning."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load pre-trained model and tokenizer
        if model_name == "all-MiniLM-L6-v2":
            # Use sentence-transformers for this model
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(model_name)
            self.tokenizer = self.sentence_transformer.tokenizer
            self.encoder = self.sentence_transformer._first_module().auto_model
        else:
            # Use regular transformers for other models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
        
        # Projection head for contrastive learning (optional)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss()
        
    def encode_sentences(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences to embeddings."""
        self.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                device = next(self.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.encoder(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def forward(self, sentences1: List[str], sentences2: List[str], labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass for training.
        
        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            labels: Similarity labels (optional, for training)
            
        Returns:
            Dictionary with embeddings and loss (if training)
        """
        # Tokenize first sentences
        inputs1 = self.tokenizer(
            sentences1,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Tokenize second sentences
        inputs2 = self.tokenizer(
            sentences2,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        # Get embeddings for first sentences
        outputs1 = self.encoder(**inputs1)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Get embeddings for second sentences
        outputs2 = self.encoder(**inputs2)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Apply projection head
        projected_embeddings1 = self.projection(embeddings1)
        projected_embeddings2 = self.projection(embeddings2)
        
        result = {
            'embeddings1': embeddings1,
            'embeddings2': embeddings2,
            'projected_embeddings1': projected_embeddings1,
            'projected_embeddings2': projected_embeddings2
        }
        
        # Compute loss if labels provided
        if labels is not None:
            labels = labels.to(device)
            loss = self.contrastive_loss(projected_embeddings1, projected_embeddings2, labels)
            result['loss'] = loss
            
        return result

class ContrastiveDataset:
    """Dataset for contrastive learning using cross-book similar pairs."""
    
    def __init__(self, similar_pairs: List[Dict], sentences: List[str], 
                 negative_sampling_ratio: int = 3):
        self.similar_pairs = similar_pairs
        self.sentences = sentences
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Create sentence to index mapping
        self.sentence_to_idx = {sent: idx for idx, sent in enumerate(sentences)}
        
        # Create training pairs
        self.training_pairs = self._create_training_pairs()
        
    def _create_training_pairs(self) -> List[Tuple[str, str, int]]:
        """Create positive and negative training pairs."""
        pairs = []
        
        # Add positive pairs from similar pairs
        for pair in self.similar_pairs:
            sent1, sent2 = pair['sentence1'], pair['sentence2']
            pairs.append((sent1, sent2, 1))  # 1 for similar
        
        # Add negative pairs (random sentences from different books)
        num_negative = len(pairs) * self.negative_sampling_ratio
        
        for _ in range(num_negative):
            # Random sentence 1
            sent1 = np.random.choice(self.sentences)
            
            # Random sentence 2 (different from sent1)
            sent2 = np.random.choice(self.sentences)
            while sent2 == sent1:
                sent2 = np.random.choice(self.sentences)
            
            pairs.append((sent1, sent2, 0))  # 0 for dissimilar
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return self.training_pairs[idx]

def create_contrastive_dataloader(similar_pairs: List[Dict], sentences: List[str], 
                                batch_size: int = 32, negative_sampling_ratio: int = 3):
    """Create dataloader for contrastive learning."""
    from torch.utils.data import DataLoader
    
    dataset = ContrastiveDataset(similar_pairs, sentences, negative_sampling_ratio)
    
    def collate_fn(batch):
        sentences1, sentences2, labels = zip(*batch)
        return list(sentences1), list(sentences2), torch.tensor(labels, dtype=torch.long)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def train_semantic_embedding_model(similar_pairs: List[Dict], sentences: List[str], 
                                 config: Dict, device: str = "auto") -> SemanticEmbeddingModel:
    """
    Train semantic embedding model using contrastive learning.
    
    Args:
        similar_pairs: List of cross-book similar pairs
        sentences: All sentences in the dataset
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained semantic embedding model
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    logger.info(f"Training on device: {device}")
    
    # Initialize model
    model = SemanticEmbeddingModel(
        model_name=config['model']['encoder']['model_name'],
        embedding_dim=config['model']['encoder']['hidden_size']
    )
    model.to(device)
    
    # Create dataloader
    dataloader = create_contrastive_dataloader(
        similar_pairs=similar_pairs,
        sentences=sentences,
        batch_size=config['training']['batch_size'],
        negative_sampling_ratio=config['model']['semantic_embedding']['contrastive_learning']['negative_sampling_ratio']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['model']['training_phases'][0]['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Training loop
    num_epochs = config['model']['training_phases'][0]['epochs']
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (sentences1, sentences2, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Forward pass
            outputs = model(sentences1, sentences2, labels)
            loss = outputs['loss']
            
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
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    # Test the model
    import json
    import yaml
    
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load semantic analysis data
    with open("data/semantic_analysis_data.json", 'r') as f:
        semantic_data = json.load(f)
    
    # Train model
    model = train_semantic_embedding_model(
        similar_pairs=semantic_data['similar_pairs'],
        sentences=[signal['sentence'] for signal in semantic_data['training_signals']],
        config=config
    )
    
    print("Model training completed!") 