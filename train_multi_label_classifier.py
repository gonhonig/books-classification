#!/usr/bin/env python3
"""
Train multi-label classifier for book sentence classification.
"""

import torch
import json
import yaml
import argparse
import sys
from pathlib import Path
import logging
from models.semantic_embedding_model import SemanticEmbeddingModel
from models.multi_label_classifier import train_multi_label_classifier, evaluate_multi_label_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def train_multi_label_classifier_model(device="auto", config_path="configs/config.yaml", semantic_model="improved", output_dir="experiments/multi_label_classifier"):
    """Train multi-label classifier."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    with open("data/semantic_analysis_data.json", 'r') as f:
        semantic_data = json.load(f)
    
    # Load metadata
    with open("data/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Determine semantic model path
    semantic_model_path = "experiments/semantic_embedding/semantic_embedding_model.pt"
    
    if not Path(semantic_model_path).exists():
        logger.error(f"Semantic model not found at {semantic_model_path}. Please train the semantic model first.")
        return False
    
    # Load semantic model
    device = "cpu"  # Use CPU for loading
    semantic_model = load_semantic_model(semantic_model_path, config, device)
    logger.info(f"{semantic_model.capitalize()} semantic model loaded successfully!")
    
    # Get training data
    sentences = [signal['sentence'] for signal in semantic_data['training_signals']]
    book_labels = [signal['original_book'] for signal in semantic_data['training_signals']]
    book_to_id = metadata['label_to_id']
    
    # Split data (use same splits as original)
    from datasets import load_from_disk
    dataset = load_from_disk("data/processed_dataset")
    
    train_sentences = dataset['train']['sentence']
    train_labels = [metadata['id_to_label'][str(label)] for label in dataset['train']['label']]
    
    val_sentences = dataset['validation']['sentence']
    val_labels = [metadata['id_to_label'][str(label)] for label in dataset['validation']['label']]
    
    test_sentences = dataset['test']['sentence']
    test_labels = [metadata['id_to_label'][str(label)] for label in dataset['test']['label']]
    
    logger.info(f"Training data: {len(train_sentences)} sentences")
    logger.info(f"Validation data: {len(val_sentences)} sentences")
    logger.info(f"Test data: {len(test_sentences)} sentences")
    
    # Train multi-label classifier
    model = train_multi_label_classifier(
        semantic_model=semantic_model,
        sentences=train_sentences,
        book_labels=train_labels,
        book_to_id=book_to_id,
        config=config,
        device=device
    )
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_results = evaluate_multi_label_model(
        model=model,
        test_sentences=val_sentences,
        test_labels=val_labels,
        book_to_id=book_to_id
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluate_multi_label_model(
        model=model,
        test_sentences=test_sentences,
        test_labels=test_labels,
        book_to_id=book_to_id
    )
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'semantic_model_path': semantic_model_path,
        'book_to_id': book_to_id,
        'val_results': val_results,
        'test_results': test_results,
        'config': config
    }
    
    torch.save(checkpoint, output_path / "multi_label_classifier.pt")
    
    # Print results
    print(f"\n=== MULTI-LABEL CLASSIFIER TRAINING COMPLETED ===")
    print(f"Model saved to: {output_path / 'multi_label_classifier.pt'}")
    print(f"Semantic model used: {semantic_model}")
    
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Overall accuracy: {val_results['overall_accuracy']:.3f}")
    print(f"Average predictions per sentence: {val_results['avg_predictions_per_sentence']:.2f}")
    print("Per-book accuracy:")
    for book, accuracy in val_results['book_metrics'].items():
        print(f"  {book}: {accuracy:.3f}")
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Overall accuracy: {test_results['overall_accuracy']:.3f}")
    print(f"Average predictions per sentence: {test_results['avg_predictions_per_sentence']:.2f}")
    print("Per-book accuracy:")
    for book, accuracy in test_results['book_metrics'].items():
        print(f"  {book}: {accuracy:.3f}")
    
    # Test some example predictions
    logger.info("Testing example predictions...")
    test_sentences = [
        "What could I do?",
        "The monster approached slowly.",
        "Alice fell down the rabbit hole.",
        "Caesar crossed the Rubicon."
    ]
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_sentences)
        book_scores = outputs['book_scores']
        
        print(f"\n=== EXAMPLE PREDICTIONS ===")
        for i, sentence in enumerate(test_sentences):
            scores = book_scores[i]
            predictions = (scores > 0.5).float()
            
            print(f"'{sentence}':")
            for book_name, book_id in book_to_id.items():
                score = scores[book_id].item()
                predicted = predictions[book_id].item()
                print(f"  {book_name}: {score:.3f} {'✓' if predicted else '✗'}")
            print()
    
    return True

def main():
    """Train multi-label classifier."""
    parser = argparse.ArgumentParser(description="Train multi-label classifier")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--semantic-model", default="improved",
                       choices=["original", "improved"],
                       help="Which semantic model to use")
    parser.add_argument("--output-dir", default="experiments/multi_label_classifier",
                       help="Output directory for model and logs")
    
    args = parser.parse_args()
    
    success = train_multi_label_classifier_model(
        device=args.device,
        config_path=args.config,
        semantic_model=args.semantic_model,
        output_dir=args.output_dir
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 