#!/usr/bin/env python3
"""
Semantic Augmented Dataset Analysis Script

This script analyzes the semantic augmented dataset and creates visualizations
of the embeddings space using PCA, colored by original book.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_semantic_dataset():
    """Load the semantic augmented dataset, excluding naive_ columns."""
    print("Loading semantic augmented dataset...")
    
    # Load the dataset
    df = pd.read_csv('../data/semantic_augmented/semantic_augmented_balanced_dataset.csv')
    
    # Get columns that don't start with 'naive_'
    non_naive_columns = [col for col in df.columns if not col.startswith('naive_')]
    df_clean = df[non_naive_columns].copy()
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    return df_clean

def load_cached_embeddings():
    """Load the cached embeddings."""
    print("Loading cached embeddings...")
    
    # Try the aligned embeddings first
    try:
        embeddings_data = np.load('../data/embeddings_cache_aligned_f24a423ed8f9dd531230fe64f71f668d.npz')
        embeddings = embeddings_data['embeddings']
        print(f"Loaded aligned embeddings with shape: {embeddings.shape}")
        return embeddings
    except:
        # Fallback to the other embeddings file
        try:
            embeddings_data = np.load('../data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')
            embeddings = embeddings_data['embeddings']
            print(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None

def comprehensive_dataset_analysis(df):
    """Perform comprehensive dataset analysis."""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("="*60)
    
    analysis_results = {}
    
    # Basic statistics
    analysis_results['basic_stats'] = {
        'total_samples': len(df),
        'total_columns': len(df.columns),
        'sentence_length_stats': {
            'mean': df['sentence'].str.len().mean(),
            'median': df['sentence'].str.len().median(),
            'std': df['sentence'].str.len().std(),
            'min': df['sentence'].str.len().min(),
            'max': df['sentence'].str.len().max()
        }
    }
    
    # Original book analysis
    book_counts = df['original_book'].value_counts()
    analysis_results['book_analysis'] = {
        'book_distribution': book_counts.to_dict(),
        'book_percentages': (book_counts / len(df) * 100).to_dict()
    }
    
    # Multi-label analysis
    book_columns = [col for col in df.columns if col.startswith('book_')]
    multi_label_stats = {}
    for col in book_columns:
        positive_count = df[col].sum()
        multi_label_stats[col] = {
            'positive_samples': int(positive_count),
            'negative_samples': int(len(df) - positive_count),
            'positive_percentage': float(positive_count / len(df) * 100)
        }
    
    analysis_results['multi_label_analysis'] = {
        'book_columns': book_columns,
        'book_statistics': multi_label_stats
    }
    
    # Multi-label patterns
    book_cols = [col for col in df.columns if col.startswith('book_')]
    df['label_count'] = df[book_cols].sum(axis=1)
    label_count_dist = df['label_count'].value_counts().sort_index()
    analysis_results['multi_label_patterns'] = {
        'label_count_distribution': label_count_dist.to_dict(),
        'label_count_percentages': (label_count_dist / len(df) * 100).to_dict()
    }
    
    # Print summary
    print(f"Total samples: {analysis_results['basic_stats']['total_samples']}")
    print(f"Average sentence length: {analysis_results['basic_stats']['sentence_length_stats']['mean']:.1f} characters")
    print(f"Number of books: {len(analysis_results['book_analysis']['book_distribution'])}")
    
    return analysis_results

def embedding_quality_analysis(embeddings, df):
    """Comprehensive embedding quality analysis."""
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    
    quality_results = {}
    
    # Basic embedding stats
    quality_results['embedding_stats'] = {
        'dimensions': embeddings.shape[1],
        'samples': embeddings.shape[0],
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'min_norm': float(np.min(np.linalg.norm(embeddings, axis=1))),
        'max_norm': float(np.max(np.linalg.norm(embeddings, axis=1)))
    }
    
    # Book separability analysis
    unique_books = sorted(df['original_book'].unique())
    separability_analysis = {}
    
    for book in unique_books:
        mask = df['original_book'] == book
        book_embeddings = embeddings[mask]
        
        if len(book_embeddings) > 1:
            # Within-book distances
            within_distances = pairwise_distances(book_embeddings, metric='euclidean')
            within_distances = within_distances[np.triu_indices_from(within_distances, k=1)]
            avg_within = np.mean(within_distances)
            std_within = np.std(within_distances)
            
            # Between-book distances
            other_mask = df['original_book'] != book
            other_embeddings = embeddings[other_mask]
            
            if len(other_embeddings) > 0:
                between_distances = pairwise_distances(book_embeddings, other_embeddings, metric='euclidean')
                avg_between = np.mean(between_distances)
                std_between = np.std(between_distances)
                
                separability_analysis[book] = {
                    'avg_within_distance': float(avg_within),
                    'std_within_distance': float(std_within),
                    'avg_between_distance': float(avg_between),
                    'std_between_distance': float(std_between),
                    'separation_ratio': float(avg_between / avg_within),
                    'sample_count': int(len(book_embeddings))
                }
    
    quality_results['separability_analysis'] = separability_analysis
    
    # Print summary
    print(f"Embedding dimensions: {quality_results['embedding_stats']['dimensions']}")
    print(f"Average embedding norm: {quality_results['embedding_stats']['mean_norm']:.4f}")
    
    for book in unique_books:
        if book in separability_analysis:
            sep = separability_analysis[book]
            print(f"{book}:")
            print(f"  Within-book distance: {sep['avg_within_distance']:.4f} ± {sep['std_within_distance']:.4f}")
            print(f"  Between-book distance: {sep['avg_between_distance']:.4f} ± {sep['std_between_distance']:.4f}")
            print(f"  Separation ratio: {sep['separation_ratio']:.4f}")
    
    return quality_results

def create_advanced_visualizations(embeddings, df):
    """Create advanced visualizations including t-SNE."""
    print("\n" + "="*60)
    print("CREATING ADVANCED VISUALIZATIONS")
    print("="*60)
    
    # PCA visualizations
    pca_2d = PCA(n_components=2, random_state=42)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings)
    
    pca_3d = PCA(n_components=3, random_state=42)
    embeddings_pca_3d = pca_3d.fit_transform(embeddings)
    
    # t-SNE visualization (on subset for performance)
    print("Computing t-SNE (this may take a while)...")
    subset_size = min(2000, len(embeddings))
    subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
    embeddings_subset = embeddings[subset_indices]
    df_subset = df.iloc[subset_indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_subset)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. PCA by book
    unique_books = sorted(df['original_book'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_books)))
    
    for i, book in enumerate(unique_books):
        mask = df['original_book'] == book
        axes[0, 0].scatter(embeddings_pca_2d[mask, 0], embeddings_pca_2d[mask, 1], 
                           c=[colors[i]], label=book, alpha=0.7, s=20)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].set_title('PCA by Original Book')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PCA by multi-label count
    label_count_colors = plt.cm.viridis(np.linspace(0, 1, 5))  # 0-4 labels
    for i in range(5):
        mask = df['label_count'] == i
        if mask.sum() > 0:
            axes[0, 1].scatter(embeddings_pca_2d[mask, 0], embeddings_pca_2d[mask, 1], 
                               c=[label_count_colors[i]], label=f'{i} labels', alpha=0.7, s=20)
    
    axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
    axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
    axes[0, 1].set_title('PCA by Multi-label Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. t-SNE by book
    for i, book in enumerate(unique_books):
        mask = df_subset['original_book'] == book
        axes[1, 0].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                           c=[colors[i]], label=book, alpha=0.7, s=20)
    
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].set_title('t-SNE by Original Book (Subset)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. t-SNE by multi-label count
    for i in range(5):
        mask = df_subset['label_count'] == i
        if mask.sum() > 0:
            axes[1, 1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                               c=[label_count_colors[i]], label=f'{i} labels', alpha=0.7, s=20)
    
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].set_title('t-SNE by Multi-label Count (Subset)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('semantic_dataset_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create 3D PCA plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, book in enumerate(unique_books):
        mask = df['original_book'] == book
        ax.scatter(embeddings_pca_3d[mask, 0], embeddings_pca_3d[mask, 1], embeddings_pca_3d[mask, 2],
                  c=[colors[i]], label=book, alpha=0.7, s=15)
    
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})')
    ax.set_title('3D PCA Visualization by Original Book')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('semantic_dataset_3d_pca_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'pca_2d_variance': pca_2d.explained_variance_ratio_.tolist(),
        'pca_3d_variance': pca_3d.explained_variance_ratio_.tolist(),
        'tsne_subset_size': subset_size
    }

def save_analysis_results(analysis_results, quality_results, viz_results):
    """Save all analysis results to JSON file."""
    print("\n" + "="*60)
    print("SAVING ANALYSIS RESULTS")
    print("="*60)
    
    all_results = {
        'dataset_analysis': analysis_results,
        'embedding_quality': quality_results,
        'visualization_results': viz_results,
        'summary': {
            'total_samples': analysis_results['basic_stats']['total_samples'],
            'embedding_dimensions': quality_results['embedding_stats']['dimensions'],
            'number_of_books': len(analysis_results['book_analysis']['book_distribution']),
            'average_separation_ratio': np.mean([
                quality_results['separability_analysis'][book]['separation_ratio']
                for book in quality_results['separability_analysis']
            ])
        }
    }
    
    with open('semantic_dataset_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Analysis results saved to: semantic_dataset_analysis_results.json")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total samples: {all_results['summary']['total_samples']}")
    print(f"Embedding dimensions: {all_results['summary']['embedding_dimensions']}")
    print(f"Number of books: {all_results['summary']['number_of_books']}")
    print(f"Average separation ratio: {all_results['summary']['average_separation_ratio']:.4f}")

def main():
    """Main analysis function."""
    print("Semantic Augmented Dataset Analysis (Book-focused)")
    print("="*60)
    
    # Load dataset
    df = load_semantic_dataset()
    
    # Load embeddings
    embeddings = load_cached_embeddings()
    
    if embeddings is None:
        print("Could not load embeddings. Exiting.")
        return
    
    # Check if embeddings match dataset size
    if len(embeddings) != len(df):
        print(f"Warning: Embeddings shape {embeddings.shape[0]} doesn't match dataset size {len(df)}")
        min_size = min(len(embeddings), len(df))
        embeddings = embeddings[:min_size]
        df = df.iloc[:min_size].reset_index(drop=True)
        print(f"Truncated to {min_size} samples")
    
    # Perform comprehensive analysis
    analysis_results = comprehensive_dataset_analysis(df)
    quality_results = embedding_quality_analysis(embeddings, df)
    viz_results = create_advanced_visualizations(embeddings, df)
    
    # Save results
    save_analysis_results(analysis_results, quality_results, viz_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- semantic_dataset_comprehensive_analysis.png")
    print("- semantic_dataset_3d_pca_detailed.png")
    print("- semantic_dataset_analysis_results.json")

if __name__ == "__main__":
    main() 