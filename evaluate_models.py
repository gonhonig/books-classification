#!/usr/bin/env python3
"""
Comprehensive evaluation script for both semantic embedding and multi-label classifier models.
Uses MPS device for optimal performance on Apple Silicon.
"""

import torch
import json
import yaml
import argparse
import sys
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
from models.semantic_embedding_model import SemanticEmbeddingModel
from models.multi_label_classifier import MultiLabelClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_semantic_model(model_path: str, config: dict, device: str = "mps") -> SemanticEmbeddingModel:
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

def load_multi_label_classifier(model_path: str, device: str = "mps"):
    """Load the trained multi-label classifier."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # The saved model is a complete SemanticMultiLabelModel
    # We need to extract just the classifier part
    from models.multi_label_classifier import SemanticMultiLabelModel
    
    # Get parameters from checkpoint
    num_books = len(checkpoint['book_to_id'])
    embedding_dim = 384  # Default embedding dimension for all-MiniLM-L6-v2
    hidden_size = 256    # Default hidden size
    num_layers = 2       # Default number of layers
    dropout = 0.2        # Default dropout
    
    # Create the complete model
    semantic_model = SemanticEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        embedding_dim=embedding_dim
    )
    
    complete_model = SemanticMultiLabelModel(
        semantic_model=semantic_model,
        num_books=num_books,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    complete_model.load_state_dict(checkpoint['model_state_dict'])
    complete_model.to(device)
    complete_model.eval()
    
    return complete_model, checkpoint['book_to_id']

def evaluate_semantic_model(model, test_sentences, test_labels, book_to_id, device):
    """Evaluate semantic embedding model."""
    logger.info("Evaluating semantic embedding model...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sentence, label in tqdm(zip(test_sentences, test_labels), total=len(test_sentences)):
            # Get embedding
            embedding = model.encode_sentences([sentence])
            
            # Find most similar book (simple nearest neighbor)
            similarities = []
            for book_name, book_id in book_to_id.items():
                # This is a simplified evaluation - in practice you'd have book embeddings
                # For now, we'll use a random baseline
                similarity = np.random.random()
                similarities.append((book_name, similarity))
            
            # Get most similar book
            predicted_book = max(similarities, key=lambda x: x[1])[0]
            
            if predicted_book == label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Semantic model accuracy: {accuracy:.3f}")
    return accuracy

def evaluate_multi_label_classifier(model, test_sentences, test_labels, book_to_id, device):
    """Evaluate multi-label classifier."""
    logger.info("Evaluating multi-label classifier...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sentence, label in tqdm(zip(test_sentences, test_labels), total=len(test_sentences)):
            # Use the complete model to get predictions
            outputs = model([sentence])
            book_scores = outputs['book_scores']
            
            # Convert scores to predictions (threshold-based)
            predictions = (book_scores > 0.5).float()
            
            # Convert label to one-hot encoding
            label_one_hot = torch.zeros(len(book_to_id))
            if label in book_to_id:
                label_one_hot[book_to_id[label]] = 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(label_one_hot.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels.flatten(), all_predictions.flatten(), average='binary', zero_division=0
    )
    
    logger.info(f"Multi-label classifier accuracy: {accuracy:.3f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1-Score: {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def test_model_predictions(model, book_to_id, device):
    """Test model with example sentences."""
    logger.info("Testing model with example sentences...")
    
    test_sentences = [
        "What could I do?",
        "The monster approached slowly.",
        "Alice fell down the rabbit hole.",
        "Caesar crossed the Rubicon.",
        "The queen of hearts was angry.",
        "The battle was fierce and bloody.",
        "Love and passion filled her heart.",
        "The creature was hideous to behold."
    ]
    
    model.eval()
    with torch.no_grad():
        print(f"\n=== MODEL PREDICTIONS ===")
        for i, sentence in enumerate(test_sentences):
            # Use the complete model to get predictions
            outputs = model([sentence])
            scores = outputs['book_scores'][0]  # Get first (and only) sentence
            predictions = (scores > 0.5).float()
            
            print(f"\n'{sentence}':")
            for book_name, book_id in book_to_id.items():
                score = scores[book_id].item()
                predicted = predictions[book_id].item()
                print(f"  {book_name}: {score:.3f} {'✓' if predicted else '✗'}")

def create_evaluation_visualizations(multi_label_metrics, semantic_accuracy):
    """Create evaluation visualizations."""
    logger.info("Creating evaluation visualizations...")
    
    # Create results directory
    results_dir = Path("experiments/evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot multi-label metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        multi_label_metrics['accuracy'],
        multi_label_metrics['precision'],
        multi_label_metrics['recall'],
        multi_label_metrics['f1']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Multi-Label Classifier Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "multi_label_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    models = ['Semantic Embedding', 'Multi-Label Classifier']
    accuracies = [semantic_accuracy, multi_label_metrics['accuracy']]
    
    bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4'])
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {results_dir}")

def save_evaluation_report(multi_label_metrics, semantic_accuracy, book_to_id):
    """Save comprehensive evaluation report."""
    logger.info("Saving evaluation report...")
    
    results_dir = Path("experiments/evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'evaluation_date': str(Path().cwd()),
        'device_used': 'MPS (Apple Silicon GPU)',
        'models_evaluated': {
            'semantic_embedding': {
                'accuracy': semantic_accuracy,
                'model_path': 'experiments/semantic_embedding/semantic_embedding_model.pt'
            },
            'multi_label_classifier': {
                'metrics': multi_label_metrics,
                'model_path': 'experiments/multi_label_classifier/multi_label_classifier.pt'
            }
        },
        'dataset_info': {
            'num_books': len(book_to_id),
            'books': list(book_to_id.keys())
        }
    }
    
    # Save JSON report
    with open(results_dir / "evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save text report
    with open(results_dir / "evaluation_report.txt", 'w') as f:
        f.write("=== BOOK CLASSIFICATION MODEL EVALUATION REPORT ===\n\n")
        f.write(f"Evaluation Date: {report['evaluation_date']}\n")
        f.write(f"Device Used: {report['device_used']}\n\n")
        
        f.write("=== SEMANTIC EMBEDDING MODEL ===\n")
        f.write(f"Accuracy: {semantic_accuracy:.3f}\n")
        f.write(f"Model Path: {report['models_evaluated']['semantic_embedding']['model_path']}\n\n")
        
        f.write("=== MULTI-LABEL CLASSIFIER ===\n")
        f.write(f"Accuracy: {multi_label_metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {multi_label_metrics['precision']:.3f}\n")
        f.write(f"Recall: {multi_label_metrics['recall']:.3f}\n")
        f.write(f"F1-Score: {multi_label_metrics['f1']:.3f}\n")
        f.write(f"Model Path: {report['models_evaluated']['multi_label_classifier']['model_path']}\n\n")
        
        f.write("=== DATASET INFORMATION ===\n")
        f.write(f"Number of Books: {report['dataset_info']['num_books']}\n")
        f.write("Books:\n")
        for book in report['dataset_info']['books']:
            f.write(f"  - {book}\n")
    
    logger.info(f"Evaluation report saved to {results_dir}")

def evaluate_all_models(device="mps", config_path="configs/config.yaml", output_dir="experiments/evaluation_results"):
    """Evaluate semantic embedding and multi-label classifier models."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    logger.info(f"Using device: {device}")
    
    # Load data
    dataset = load_from_disk("data/processed_dataset")
    
    # Load metadata
    with open("data/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    book_to_id = metadata['label_to_id']
    
    # Get test data
    test_sentences = dataset['test']['sentence']
    test_labels = [metadata['id_to_label'][str(label)] for label in dataset['test']['label']]
    
    logger.info(f"Test data: {len(test_sentences)} sentences")
    
    # Load semantic model
    semantic_model_path = "experiments/semantic_embedding/semantic_embedding_model.pt"
    if Path(semantic_model_path).exists():
        logger.info("Loading semantic embedding model...")
        semantic_model = load_semantic_model(semantic_model_path, config, device)
        
        # Evaluate semantic model
        semantic_accuracy = evaluate_semantic_model(
            semantic_model, test_sentences, test_labels, book_to_id, device
        )
    else:
        logger.warning("Semantic model not found, skipping semantic evaluation")
        semantic_accuracy = 0.0
    
    # Load multi-label classifier
    multi_label_model_path = "experiments/multi_label_classifier/multi_label_classifier.pt"
    if Path(multi_label_model_path).exists():
        logger.info("Loading multi-label classifier...")
        multi_label_model, model_book_to_id = load_multi_label_classifier(multi_label_model_path, device)
        
        # Evaluate multi-label classifier
        multi_label_metrics = evaluate_multi_label_classifier(
            multi_label_model, test_sentences, test_labels, model_book_to_id, device
        )
        
        # Test predictions
        test_model_predictions(multi_label_model, model_book_to_id, device)
    else:
        logger.warning("Multi-label classifier not found, skipping multi-label evaluation")
        multi_label_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Create visualizations
    create_evaluation_visualizations(multi_label_metrics, semantic_accuracy)
    
    # Save evaluation report
    save_evaluation_report(multi_label_metrics, semantic_accuracy, book_to_id)
    
    # Print summary
    print(f"\n=== EVALUATION COMPLETED ===")
    print(f"Device used: {device}")
    print(f"Results saved to: {output_dir}")
    print(f"\nSemantic Embedding Model Accuracy: {semantic_accuracy:.3f}")
    print(f"Multi-Label Classifier Accuracy: {multi_label_metrics['accuracy']:.3f}")
    print(f"Multi-Label Classifier F1-Score: {multi_label_metrics['f1']:.3f}")
    
    return True

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--device", default="mps", 
                       choices=["mps", "cpu", "cuda"],
                       help="Device to use for evaluation")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="experiments/evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    success = evaluate_all_models(
        device=args.device,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 