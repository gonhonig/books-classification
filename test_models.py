#!/usr/bin/env python3
"""
Comprehensive testing script for the trained models.
Demonstrates model capabilities with various example sentences and provides detailed analysis.
"""

import torch
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
from models.semantic_embedding_model import SemanticEmbeddingModel
from models.multi_label_classifier import SemanticMultiLabelModel
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = "mps"):
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
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

def test_with_example_sentences(model, book_to_id, device):
    """Test the model with carefully crafted example sentences."""
    logger.info("Testing model with example sentences...")
    
    # Create reverse mapping from id to book name
    id_to_book = {v: k for k, v in book_to_id.items()}
    
    # Example sentences designed to test different aspects
    test_sentences = [
        # Frankenstein-related sentences
        "The monster was hideous to behold.",
        "Victor Frankenstein created life from death.",
        "The creature wandered alone in the wilderness.",
        "Lightning struck the laboratory.",
        "The monster sought revenge on his creator.",
        
        # Alice in Wonderland sentences
        "Alice fell down the rabbit hole.",
        "The queen of hearts was very angry.",
        "The Cheshire cat smiled mysteriously.",
        "Alice grew taller and shorter.",
        "The Mad Hatter had a tea party.",
        
        # Julius Caesar sentences
        "Caesar crossed the Rubicon river.",
        "The Ides of March came and went.",
        "Brutus was an honorable man.",
        "The senate was filled with conspirators.",
        "The battle was fierce and bloody.",
        
        # Anna Karenina sentences
        "Love and passion filled her heart.",
        "The train station was crowded.",
        "Society frowned upon their affair.",
        "The ball was elegant and refined.",
        "Tragedy struck the family.",
        
        # Mixed/ambiguous sentences
        "The creature was lonely and misunderstood.",
        "She fell into a deep sleep.",
        "The battle raged on for days.",
        "Love conquered all obstacles.",
        "The monster found acceptance at last."
    ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for sentence in test_sentences:
            outputs = model([sentence])
            book_scores = outputs['book_scores'][0]
            
            # Get predictions and confidence scores
            predictions = (book_scores > 0.5).float()
            confidence_scores = book_scores.cpu().numpy()
            
            # Find the most confident prediction
            max_score_idx = np.argmax(confidence_scores)
            max_score = confidence_scores[max_score_idx]
            predicted_book = id_to_book[max_score_idx]
            
            # Get all scores above threshold
            above_threshold = confidence_scores > 0.5
            predicted_books = [id_to_book[i] for i in range(len(confidence_scores)) if above_threshold[i]]
            
            results.append({
                'sentence': sentence,
                'predicted_book': predicted_book,
                'confidence': float(max_score),
                'all_predictions': predicted_books,
                'all_scores': confidence_scores.tolist()
            })
    
    return results

def analyze_predictions(results, book_to_id):
    """Analyze the prediction results."""
    logger.info("Analyzing predictions...")
    
    # Create reverse mapping
    id_to_book = {v: k for k, v in book_to_id.items()}
    
    # Count predictions per book
    book_counts = {}
    for result in results:
        book = result['predicted_book']
        book_counts[book] = book_counts.get(book, 0) + 1
    
    # Calculate average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    # Find sentences with multiple predictions
    multi_predictions = [r for r in results if len(r['all_predictions']) > 1]
    
    # Find highest confidence predictions
    high_confidence = [r for r in results if r['confidence'] > 0.7]
    
    analysis = {
        'total_sentences': len(results),
        'book_distribution': book_counts,
        'average_confidence': avg_confidence,
        'multi_predictions_count': len(multi_predictions),
        'high_confidence_count': len(high_confidence),
        'confidence_range': {
            'min': min([r['confidence'] for r in results]),
            'max': max([r['confidence'] for r in results])
        }
    }
    
    return analysis

def create_visualizations(results, analysis, save_dir: str = "experiments/test_results"):
    """Create visualizations of the test results."""
    logger.info("Creating visualizations...")
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 1. Book distribution pie chart
    plt.figure(figsize=(10, 8))
    books = list(analysis['book_distribution'].keys())
    counts = list(analysis['book_distribution'].values())
    
    plt.pie(counts, labels=books, autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution by Book')
    plt.axis('equal')
    plt.savefig(save_path / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence scores histogram
    plt.figure(figsize=(10, 6))
    confidences = [r['confidence'] for r in results]
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.title('Distribution of Confidence Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence scores by book
    plt.figure(figsize=(12, 8))
    book_confidences = {}
    for result in results:
        book = result['predicted_book']
        if book not in book_confidences:
            book_confidences[book] = []
        book_confidences[book].append(result['confidence'])
    
    books = list(book_confidences.keys())
    confidences = [book_confidences[book] for book in books]
    
    plt.boxplot(confidences, labels=books)
    plt.xlabel('Book')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Score Distribution by Book')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'confidence_by_book.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {save_path}")

def print_detailed_results(results, analysis):
    """Print detailed test results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TESTING RESULTS")
    print("="*80)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   • Total test sentences: {analysis['total_sentences']}")
    print(f"   • Average confidence: {analysis['average_confidence']:.3f}")
    print(f"   • Confidence range: {analysis['confidence_range']['min']:.3f} - {analysis['confidence_range']['max']:.3f}")
    print(f"   • High confidence predictions (>0.7): {analysis['high_confidence_count']}")
    print(f"   • Multi-predictions: {analysis['multi_predictions_count']}")
    
    print(f"\n📚 PREDICTION DISTRIBUTION:")
    for book, count in analysis['book_distribution'].items():
        percentage = (count / analysis['total_sentences']) * 100
        print(f"   • {book}: {count} predictions ({percentage:.1f}%)")
    
    print(f"\n🔍 DETAILED PREDICTIONS:")
    for i, result in enumerate(results, 1):
        print(f"\n{i:2d}. '{result['sentence']}'")
        print(f"    Predicted: {result['predicted_book']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        if len(result['all_predictions']) > 1:
            print(f"    All predictions: {', '.join(result['all_predictions'])}")
    
    # Show highest confidence predictions
    high_conf = sorted(results, key=lambda x: x['confidence'], reverse=True)[:5]
    print(f"\n🏆 HIGHEST CONFIDENCE PREDICTIONS:")
    for i, result in enumerate(high_conf, 1):
        print(f"   {i}. '{result['sentence']}' → {result['predicted_book']} ({result['confidence']:.3f})")
    
    # Show multi-predictions
    multi_pred = [r for r in results if len(r['all_predictions']) > 1]
    if multi_pred:
        print(f"\n🤔 MULTI-PREDICTIONS (Ambiguous cases):")
        for i, result in enumerate(multi_pred, 1):
            print(f"   {i}. '{result['sentence']}'")
            print(f"      Predictions: {', '.join(result['all_predictions'])}")

def save_test_results(results, analysis, save_path: str = "experiments/test_results"):
    """Save test results to files."""
    logger.info("Saving test results...")
    
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    output = {
        'results': results,
        'analysis': analysis,
        'timestamp': str(Path().cwd()),
        'model_info': {
            'total_books': len(analysis['book_distribution']),
            'test_sentences': len(results)
        }
    }
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save summary report
    with open(save_dir / 'test_summary.txt', 'w') as f:
        f.write("MODEL TESTING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total test sentences: {analysis['total_sentences']}\n")
        f.write(f"Average confidence: {analysis['average_confidence']:.3f}\n")
        f.write(f"High confidence predictions: {analysis['high_confidence_count']}\n")
        f.write(f"Multi-predictions: {analysis['multi_predictions_count']}\n\n")
        
        f.write("PREDICTION DISTRIBUTION:\n")
        for book, count in analysis['book_distribution'].items():
            percentage = (count / analysis['total_sentences']) * 100
            f.write(f"  {book}: {count} ({percentage:.1f}%)\n")
    
    logger.info(f"Test results saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test trained models with example sentences")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--model-path", type=str, 
                       default="experiments/fine_tuned_model.pt",
                       help="Path to trained model")
    parser.add_argument("--save-dir", type=str, 
                       default="experiments/test_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading model from: {args.model_path}")
    
    # Load model
    model, book_to_id = load_model(args.model_path, args.device)
    logger.info(f"Model loaded successfully. Number of books: {len(book_to_id)}")
    
    # Test with example sentences
    results = test_with_example_sentences(model, book_to_id, args.device)
    
    # Analyze results
    analysis = analyze_predictions(results, book_to_id)
    
    # Create visualizations
    create_visualizations(results, analysis, args.save_dir)
    
    # Print detailed results
    print_detailed_results(results, analysis)
    
    # Save results
    save_test_results(results, analysis, args.save_dir)
    
    logger.info("=== TESTING COMPLETED ===")
    logger.info(f"Device used: {args.device}")
    logger.info(f"Results saved to: {args.save_dir}")
    logger.info(f"Average confidence: {analysis['average_confidence']:.3f}")

if __name__ == "__main__":
    main() 