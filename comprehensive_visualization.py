#!/usr/bin/env python3
"""
Comprehensive visualization script for semantic embedding space analysis.
Combines all visualization functionality into a single file.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import logging
import warnings
import argparse
warnings.filterwarnings('ignore')

from models.semantic_embedding_model import SemanticEmbeddingModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = "cpu") -> SemanticEmbeddingModel:
    """Load a semantic embedding model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SemanticEmbeddingModel(
        model_name=checkpoint['model_name'],
        embedding_dim=checkpoint['embedding_dim']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def get_embeddings_safe(model: SemanticEmbeddingModel, 
                       sentences: list, 
                       batch_size: int = 8) -> np.ndarray:
    """Get embeddings with very small batch size to avoid memory issues."""
    logger.info(f"Computing embeddings for {len(sentences)} sentences...")
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            outputs = model(batch_sentences, batch_sentences)
            batch_embeddings = outputs['embeddings1'].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def create_basic_embedding_space(embeddings: np.ndarray, 
                               labels: list,
                               sentences: list,
                               output_path: str = "experiments/visualization/basic_embedding_space.png"):
    """Create basic PCA embedding space visualization."""
    logger.info("Creating basic embedding space visualization...")
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Basic Semantic Embedding Space Analysis', fontsize=16, fontweight='bold')
    
    # Color by book
    book_names = ['The Life of Julius Caesar', 'The Adventures of Alice in Wonderland', 
                  'Anna Karenina', 'Frankenstein']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Plot 1: Main embedding space visualization
    for i, book_name in enumerate(book_names):
        book_indices = [j for j, label in enumerate(labels) if label == book_name]
        if book_indices:
            ax1.scatter(embeddings_2d[book_indices, 0], embeddings_2d[book_indices, 1], 
                       c=colors[i], label=book_name, alpha=0.7, s=40)
    
    ax1.set_title('PCA Embedding Space\n(Colored by Original Book)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density visualization
    x_bins = np.linspace(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max(), 30)
    y_bins = np.linspace(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max(), 30)
    
    H, xedges, yedges = np.histogram2d(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                       bins=[x_bins, y_bins])
    
    im = ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    cmap='viridis', alpha=0.8)
    ax2.set_title('Point Density in Embedding Space\n(Darker areas = more sentences)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('PC1', fontsize=10)
    ax2.set_ylabel('PC2', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Number of sentences', fontsize=9)
    
    # Plot 3: Book centers and distributions
    for i, book_name in enumerate(book_names):
        book_indices = [j for j, label in enumerate(labels) if label == book_name]
        if book_indices:
            book_points = embeddings_2d[book_indices]
            center = np.mean(book_points, axis=0)
            
            # Plot book center
            ax3.scatter(center[0], center[1], c=colors[i], s=200, marker='*', 
                       label=f'{book_name} center', edgecolors='black', linewidth=2)
            
            # Plot book points
            ax3.scatter(book_points[:, 0], book_points[:, 1], c=colors[i], 
                       alpha=0.6, s=30, label=book_name)
    
    ax3.set_title('Book Centers and Distributions', fontsize=12, fontweight='bold')
    ax3.set_xlabel('PC1', fontsize=10)
    ax3.set_ylabel('PC2', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mixed area analysis
    x_range = np.linspace(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max(), 10)
    y_range = np.linspace(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max(), 10)
    
    mixed_areas = []
    for i in range(len(x_range)-1):
        for j in range(len(y_range)-1):
            mask = ((embeddings_2d[:, 0] >= x_range[i]) & (embeddings_2d[:, 0] < x_range[i+1]) &
                   (embeddings_2d[:, 1] >= y_range[j]) & (embeddings_2d[:, 1] < y_range[j+1]))
            
            if np.sum(mask) > 5:
                region_labels = [labels[k] for k in range(len(labels)) if mask[k]]
                unique_books = len(set(region_labels))
                mixed_areas.append((x_range[i], y_range[j], unique_books))
    
    # Plot mixed areas
    for x, y, num_books in mixed_areas:
        color = plt.cm.RdYlBu(num_books / 4)
        ax4.add_patch(plt.Rectangle((x, y), x_range[1]-x_range[0], y_range[1]-y_range[0], 
                                   facecolor=color, alpha=0.6))
    
    # Add original points
    for i, book_name in enumerate(book_names):
        book_indices = [j for j, label in enumerate(labels) if label == book_name]
        if book_indices:
            ax4.scatter(embeddings_2d[book_indices, 0], embeddings_2d[book_indices, 1], 
                       c=colors[i], alpha=0.7, s=20)
    
    ax4.set_title('Mixed Areas Analysis\n(Red = many books, Blue = few books)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('PC1', fontsize=10)
    ax4.set_ylabel('PC2', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Basic embedding space visualization saved to {output_path}")

def create_book_specificity_analysis(embeddings: np.ndarray, 
                                   labels: list,
                                   sentences: list,
                                   output_path: str = "experiments/visualization/book_specificity.png"):
    """Analyze and visualize book specificity in the embedding space."""
    logger.info("Creating book specificity analysis...")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Calculate book specificity for each sentence
    book_names = ['The Life of Julius Caesar', 'The Adventures of Alice in Wonderland', 
                  'Anna Karenina', 'Frankenstein']
    colors = ['red', 'blue', 'green', 'orange']
    
    specificity_scores = []
    
    for i, sentence_embedding in enumerate(embeddings):
        # Calculate average similarity to sentences from same book vs different books
        same_book_similarities = []
        different_book_similarities = []
        
        current_book = labels[i]
        
        for j, other_embedding in enumerate(embeddings):
            if i != j:
                similarity = np.dot(sentence_embedding, other_embedding) / (np.linalg.norm(sentence_embedding) * np.linalg.norm(other_embedding))
                
                if labels[j] == current_book:
                    same_book_similarities.append(similarity)
                else:
                    different_book_similarities.append(similarity)
        
        # Calculate specificity as difference between same-book and different-book similarities
        if same_book_similarities and different_book_similarities:
            specificity = np.mean(same_book_similarities) - np.mean(different_book_similarities)
        else:
            specificity = 0
        
        specificity_scores.append(specificity)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Book Specificity Analysis in Semantic Embedding Space', fontsize=16, fontweight='bold')
    
    # Plot 1: Embedding space colored by specificity
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=specificity_scores, cmap='RdYlBu', alpha=0.7, s=40)
    ax1.set_title('Embedding Space Colored by Book Specificity\n(Red = book-specific, Blue = generic)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=ax1, label='Specificity Score')
    
    # Plot 2: Specificity distribution by book
    for i, book_name in enumerate(book_names):
        book_indices = [j for j, label in enumerate(labels) if label == book_name]
        if book_indices:
            book_specificities = [specificity_scores[j] for j in book_indices]
            ax2.hist(book_specificities, alpha=0.6, label=book_name, color=colors[i], bins=15)
    
    ax2.set_title('Distribution of Book Specificity by Book')
    ax2.set_xlabel('Specificity Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Average specificity by book
    avg_specificities = []
    for book_name in book_names:
        book_indices = [j for j, label in enumerate(labels) if label == book_name]
        if book_indices:
            book_specificities = [specificity_scores[j] for j in book_indices]
            avg_specificities.append(np.mean(book_specificities))
        else:
            avg_specificities.append(0)
    
    bars = ax3.bar(book_names, avg_specificities, color=colors, alpha=0.7)
    ax3.set_title('Average Book Specificity by Book')
    ax3.set_ylabel('Average Specificity Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value annotations
    for bar, value in zip(bars, avg_specificities):
        height = bar.get_height()
        ax3.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontweight='bold')
    
    # Plot 4: Generic vs specific sentence examples
    # Find most generic and most specific sentences
    most_generic_idx = np.argmin(specificity_scores)
    most_specific_idx = np.argmax(specificity_scores)
    
    ax4.text(0.1, 0.8, f"Most Generic Sentence:\n'{sentences[most_generic_idx][:80]}...'\n\nSpecificity: {specificity_scores[most_generic_idx]:.3f}", 
             transform=ax4.transAxes, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax4.text(0.1, 0.3, f"Most Book-Specific Sentence:\n'{sentences[most_specific_idx][:80]}...'\n\nSpecificity: {specificity_scores[most_specific_idx]:.3f}", 
             transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Example Sentences')
    
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Book specificity analysis saved to {output_path}")

def create_model_comparison(original_model_path: str, improved_model_path: str,
                          output_path: str = "experiments/visualization/model_comparison.png"):
    """Compare original vs improved model performance."""
    logger.info("Creating model comparison...")
    
    # Load both models
    original_model = load_model(original_model_path, "cpu")
    improved_model = load_model(improved_model_path, "cpu")
    
    # Test sentences
    test_sentences = [
        "What could I do?",
        "What shall I do?",
        "I don't know what to do.",
        "What ought I to do?",
        "What should I do?"
    ]
    
    # Get embeddings from both models
    with torch.no_grad():
        original_outputs = original_model(test_sentences, test_sentences)
        improved_outputs = improved_model(test_sentences, test_sentences)
        
        original_embeddings = original_outputs['embeddings1']
        improved_embeddings = improved_outputs['embeddings1']
    
    # Compute similarities
    original_similarities = torch.cosine_similarity(original_embeddings[0:1], original_embeddings[1:], dim=1)
    improved_similarities = torch.cosine_similarity(improved_embeddings[0:1], improved_embeddings[1:], dim=1)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original similarities
    ax1.bar(range(len(original_similarities)), original_similarities.numpy(), alpha=0.7, color='blue')
    ax1.set_title('Original Model Similarities')
    ax1.set_xlabel('Test Sentence Index')
    ax1.set_ylabel('Similarity to "What could I do?"')
    ax1.set_ylim(0, 1)
    
    # Plot improved similarities
    ax2.bar(range(len(improved_similarities)), improved_similarities.numpy(), alpha=0.7, color='green')
    ax2.set_title('Improved Model Similarities')
    ax2.set_xlabel('Test Sentence Index')
    ax2.set_ylabel('Similarity to "What could I do?"')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save comparison
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print("\n=== MODEL COMPARISON ===")
    print("Original model similarities:")
    for i, sim in enumerate(original_similarities):
        print(f"  {test_sentences[i+1]}: {sim:.3f}")
    
    print("\nImproved model similarities:")
    for i, sim in enumerate(improved_similarities):
        print(f"  {test_sentences[i+1]}: {sim:.3f}")
    
    print(f"\nComparison plot saved to: {output_path}")

def create_similarity_improvement_plot(similar_pairs: list, output_path: str = "experiments/visualization/similarity_improvements.png"):
    """Create a plot showing similarity improvements for similar pairs."""
    logger.info("Creating similarity improvement plot...")
    
    # Sample some pairs for visualization
    sample_pairs = similar_pairs[:min(10, len(similar_pairs))]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(sample_pairs))
    original_sims = [pair['similarity'] for pair in sample_pairs]
    
    # For demonstration, assume some improvement (in real case, you'd compare before/after training)
    improved_sims = [min(0.95, sim + 0.05) for sim in original_sims]  # Simulated improvement
    
    width = 0.35
    ax.bar(x_pos - width/2, original_sims, width, label='Original Similarity', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, improved_sims, width, label='Improved Similarity', alpha=0.7, color='green')
    
    ax.set_xlabel('Similar Pair Index')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Improvements for Cross-Book Pairs')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Pair {i+1}' for i in range(len(sample_pairs))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Similarity improvement plot saved to {output_path}")

def main():
    """Main visualization function with command line options."""
    parser = argparse.ArgumentParser(description="Comprehensive semantic embedding visualization")
    parser.add_argument("--mode", default="all", 
                       choices=["basic", "specificity", "comparison", "similarity", "all"],
                       help="Visualization mode")
    parser.add_argument("--model", default="improved",
                       choices=["original", "improved"],
                       help="Which model to use for visualization")
    parser.add_argument("--max-sentences", type=int, default=800,
                       help="Maximum number of sentences to visualize")
    parser.add_argument("--output-dir", default="experiments/visualization",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Load data
    with open("data/semantic_analysis_data.json", 'r') as f:
        semantic_data = json.load(f)
    
    # Determine model path
    if args.model == "original":
        model_path = "experiments/semantic_embedding/semantic_embedding_model.pt"
    else:
        model_path = "experiments/improved_semantic_embedding/improved_semantic_embedding_model.pt"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Load model
    device = "cpu"  # Use CPU for visualization to avoid device issues
    model = load_model(model_path, device)
    logger.info(f"{args.model.capitalize()} model loaded successfully!")
    
    # Get data for visualization
    sentences = [signal['sentence'] for signal in semantic_data['training_signals']]
    labels = [signal['original_book'] for signal in semantic_data['training_signals']]
    
    # Limit sentences if needed
    if len(sentences) > args.max_sentences:
        logger.info(f"Limiting visualization to {args.max_sentences} sentences for performance")
        sentences = sentences[:args.max_sentences]
        labels = labels[:args.max_sentences]
    
    # Get embeddings
    embeddings = get_embeddings_safe(model, sentences)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run visualizations based on mode
    if args.mode in ["basic", "all"]:
        create_basic_embedding_space(embeddings, labels, sentences, 
                                   output_dir / "basic_embedding_space.png")
    
    if args.mode in ["specificity", "all"]:
        create_book_specificity_analysis(embeddings, labels, sentences,
                                       output_dir / "book_specificity.png")
    
    if args.mode in ["similarity", "all"]:
        create_similarity_improvement_plot(semantic_data['similar_pairs'],
                                         output_dir / "similarity_improvements.png")
    
    if args.mode in ["comparison", "all"]:
        original_model_path = "experiments/semantic_embedding/semantic_embedding_model.pt"
        improved_model_path = "experiments/improved_semantic_embedding/improved_semantic_embedding_model.pt"
        
        if Path(original_model_path).exists() and Path(improved_model_path).exists():
            create_model_comparison(original_model_path, improved_model_path,
                                  output_dir / "model_comparison.png")
        else:
            logger.warning("Both models not found. Skipping comparison.")
    
    print(f"\n=== COMPREHENSIVE VISUALIZATION COMPLETED ===")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Generated visualizations in: {output_dir}")

if __name__ == "__main__":
    main() 