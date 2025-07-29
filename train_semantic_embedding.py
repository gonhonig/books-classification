#!/usr/bin/env python3
"""
Training script for semantic embedding model using contrastive learning.
"""

import os
import json
import yaml
import logging
import argparse
from pathlib import Path
import torch

from models.semantic_embedding_model import train_semantic_embedding_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load semantic analysis data and config."""
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load semantic analysis data
    with open("data/semantic_analysis_data.json", 'r') as f:
        semantic_data = json.load(f)
    
    return config, semantic_data

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train semantic embedding model")
    parser.add_argument("--device", default="auto", 
                       help="Device to train on (auto, cpu, cuda, mps)")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="experiments/semantic_embedding",
                       help="Output directory for model checkpoints")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    config, semantic_data = load_data()
    
    # Extract training data
    similar_pairs = semantic_data['similar_pairs']
    sentences = [signal['sentence'] for signal in semantic_data['training_signals']]
    
    logger.info(f"Loaded {len(similar_pairs)} similar pairs")
    logger.info(f"Loaded {len(sentences)} sentences")
    
    # Train model
    logger.info("Starting semantic embedding training...")
    model = train_semantic_embedding_model(
        similar_pairs=similar_pairs,
        sentences=sentences,
        config=config,
        device=args.device
    )
    
    # Save model
    model_path = output_dir / "semantic_embedding_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_name': model.model_name,
        'embedding_dim': model.embedding_dim
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Test encoding
    logger.info("Testing model encoding...")
    test_sentences = sentences[:10]  # Test with first 10 sentences
    embeddings = model.encode_sentences(test_sentences)
    
    logger.info(f"Encoded {len(test_sentences)} sentences to {embeddings.shape[1]}-dimensional embeddings")
    
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Test embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main() 