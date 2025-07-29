#!/usr/bin/env python3
"""
Train semantic embedding model with improved similar pairs.
"""

import torch
import json
import argparse
import yaml
from pathlib import Path
import logging
from models.semantic_embedding_model import train_semantic_embedding_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train semantic embedding model with improved data."""
    parser = argparse.ArgumentParser(description="Train semantic embedding model with improved data")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="experiments/improved_semantic_embedding",
                       help="Output directory for model and logs")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load improved semantic analysis data
    with open("data/improved_semantic_analysis_data.json", 'r') as f:
        improved_data = json.load(f)
    
    # Extract similar pairs
    similar_pairs = improved_data['sample_similar_pairs']
    
    # Create sentences list from all pairs
    sentences = []
    for pair in similar_pairs:
        sentences.append(pair['sentence1'])
        sentences.append(pair['sentence2'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    logger.info(f"Using {len(similar_pairs)} improved similar pairs")
    logger.info(f"Total unique sentences: {len(unique_sentences)}")
    
    # Train the model
    model = train_semantic_embedding_model(
        similar_pairs=similar_pairs,
        sentences=unique_sentences,
        config=config,
        device=args.device
    )
    
    # Save the trained model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'embedding_dim': model.embedding_dim,
        'similar_pairs_count': len(similar_pairs),
        'total_sentences': len(unique_sentences)
    }
    
    torch.save(checkpoint, output_path / "improved_semantic_embedding_model.pt")
    
    print(f"\n=== IMPROVED SEMANTIC EMBEDDING TRAINING COMPLETED ===")
    print(f"Model saved to: {output_path / 'improved_semantic_embedding_model.pt'}")
    print(f"Similar pairs used: {len(similar_pairs)}")
    print(f"Unique sentences: {len(unique_sentences)}")
    
    # Test the model
    logger.info("Testing model encoding...")
    test_sentences = [
        "What could I do?",
        "What shall I do?",
        "I don't know what to do."
    ]
    
    with torch.no_grad():
        outputs = model(test_sentences, test_sentences)
        embeddings = outputs['embeddings1']
        
        # Compute similarities
        similarities = torch.cosine_similarity(embeddings[0:1], embeddings[1:], dim=1)
        logger.info(f"Similarity between 'What could I do?' and 'What shall I do?': {similarities[0]:.3f}")
        logger.info(f"Similarity between 'What could I do?' and 'I don't know what to do.': {similarities[1]:.3f}")

if __name__ == "__main__":
    main() 