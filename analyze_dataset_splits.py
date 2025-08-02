"""
Analyze label distribution across train, validation, and test splits.
"""

import json
import pandas as pd
import numpy as np
from datasets import load_from_disk
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset_splits(dataset_path: str = 'data/dataset'):
    """Load the dataset splits."""
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Convert to DataFrames for easier analysis
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    print(f"Loaded splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def analyze_label_distribution(df: pd.DataFrame, split_name: str) -> Dict:
    """Analyze label distribution for a dataset split."""
    print(f"\n📊 {split_name.upper()} SPLIT ANALYSIS:")
    
    # Count labels per sample
    label_counts = []
    for labels in df['labels']:
        label_counts.append(sum(labels))
    
    # Distribution of label counts
    label_dist = pd.Series(label_counts).value_counts().sort_index()
    print(f"Label count distribution:")
    for num_labels, count in label_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {num_labels} label(s): {count} samples ({percentage:.1f}%)")
    
    # Per-book label distribution
    books = ['Anna Karenina', 'Frankenstein', 'The Adventures of Alice in Wonderland', 'Wuthering Heights']
    book_counts = []
    for i in range(4):
        count = sum(1 for labels in df['labels'] if labels[i] == 1)
        percentage = (count / len(df)) * 100
        book_counts.append(count)
        print(f"  {books[i]}: {count} samples ({percentage:.1f}%)")
    
    # Original label distribution
    original_label_dist = df['original_label'].value_counts().sort_index()
    print(f"Original label distribution:")
    for label, count in original_label_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
    
    return {
        'split_name': split_name,
        'total_samples': len(df),
        'label_count_distribution': label_dist.to_dict(),
        'book_distribution': dict(zip(books, book_counts)),
        'original_label_distribution': original_label_dist.to_dict()
    }

def create_visualizations(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Create visualizations for the dataset splits."""
    print(f"\n📈 CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Function to get label counts
    def get_label_counts(df):
        return [sum(labels) for labels in df['labels']]
    
    # 1. Label count distribution comparison
    train_counts = get_label_counts(train_df)
    val_counts = get_label_counts(val_df)
    test_counts = get_label_counts(test_df)
    
    # Train split
    train_dist = pd.Series(train_counts).value_counts().sort_index()
    axes[0, 0].bar(train_dist.index, train_dist.values)
    axes[0, 0].set_title('Train Split - Label Count Distribution')
    axes[0, 0].set_xlabel('Number of Labels')
    axes[0, 0].set_ylabel('Number of Samples')
    
    # Validation split
    val_dist = pd.Series(val_counts).value_counts().sort_index()
    axes[0, 1].bar(val_dist.index, val_dist.values)
    axes[0, 1].set_title('Validation Split - Label Count Distribution')
    axes[0, 1].set_xlabel('Number of Labels')
    axes[0, 1].set_ylabel('Number of Samples')
    
    # Test split
    test_dist = pd.Series(test_counts).value_counts().sort_index()
    axes[0, 2].bar(test_dist.index, test_dist.values)
    axes[0, 2].set_title('Test Split - Label Count Distribution')
    axes[0, 2].set_xlabel('Number of Labels')
    axes[0, 2].set_ylabel('Number of Samples')
    
    # 2. Book distribution comparison
    books = ['Anna Karenina', 'Frankenstein', 'The Adventures of Alice in Wonderland', 'Wuthering Heights']
    
    def get_book_counts(df):
        counts = []
        for i in range(4):
            count = sum(1 for labels in df['labels'] if labels[i] == 1)
            counts.append(count)
        return counts
    
    train_book_counts = get_book_counts(train_df)
    val_book_counts = get_book_counts(val_df)
    test_book_counts = get_book_counts(test_df)
    
    x = np.arange(len(books))
    width = 0.25
    
    axes[1, 0].bar(x - width, train_book_counts, width, label='Train')
    axes[1, 0].bar(x, val_book_counts, width, label='Validation')
    axes[1, 0].bar(x + width, test_book_counts, width, label='Test')
    axes[1, 0].set_title('Book Distribution Across Splits')
    axes[1, 0].set_xlabel('Books')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([book[:15] + '...' if len(book) > 15 else book for book in books], rotation=45)
    axes[1, 0].legend()
    
    # 3. Original label distribution
    train_original = train_df['original_label'].value_counts().sort_index()
    val_original = val_df['original_label'].value_counts().sort_index()
    test_original = test_df['original_label'].value_counts().sort_index()
    
    axes[1, 1].bar(train_original.index - width, train_original.values, width, label='Train')
    axes[1, 1].bar(val_original.index, val_original.values, width, label='Validation')
    axes[1, 1].bar(test_original.index + width, test_original.values, width, label='Test')
    axes[1, 1].set_title('Original Label Distribution Across Splits')
    axes[1, 1].set_xlabel('Original Label')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].legend()
    
    # 4. Split size comparison
    split_sizes = [len(train_df), len(val_df), len(test_df)]
    split_names = ['Train', 'Validation', 'Test']
    axes[1, 2].pie(split_sizes, labels=split_names, autopct='%1.1f%%')
    axes[1, 2].set_title('Dataset Split Sizes')
    
    plt.tight_layout()
    plt.savefig('data/semantic_augmented/dataset_splits_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Visualizations saved to data/semantic_augmented/dataset_splits_analysis.png")
    
    plt.show()

def save_analysis_report(analyses: List[Dict], output_path: str = 'data/semantic_augmented/dataset_splits_analysis.json'):
    """Save detailed analysis report."""
    report = {
        'analyses': analyses,
        'summary': {
            'total_samples': sum(analysis['total_samples'] for analysis in analyses),
            'split_ratios': {
                analysis['split_name']: analysis['total_samples'] 
                for analysis in analyses
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Analysis report saved to {output_path}")

def main():
    """Main function to analyze dataset splits."""
    print("🔍 Analyzing dataset splits distribution...")
    
    # Load dataset splits
    train_df, val_df, test_df = load_dataset_splits()
    
    # Analyze each split
    analyses = []
    analyses.append(analyze_label_distribution(train_df, 'train'))
    analyses.append(analyze_label_distribution(val_df, 'validation'))
    analyses.append(analyze_label_distribution(test_df, 'test'))
    
    # Create visualizations
    create_visualizations(train_df, val_df, test_df)
    
    # Save analysis report
    save_analysis_report(analyses)
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    print(f"Total samples: {sum(analysis['total_samples'] for analysis in analyses)}")
    print(f"Split ratios: Train={analyses[0]['total_samples']}, Val={analyses[1]['total_samples']}, Test={analyses[2]['total_samples']}")
    
    # Check distribution consistency
    print(f"\n✅ DISTRIBUTION CONSISTENCY CHECK:")
    for i in range(1, 5):  # Check label counts 1-4
        train_pct = analyses[0]['label_count_distribution'].get(i, 0) / analyses[0]['total_samples'] * 100
        val_pct = analyses[1]['label_count_distribution'].get(i, 0) / analyses[1]['total_samples'] * 100
        test_pct = analyses[2]['label_count_distribution'].get(i, 0) / analyses[2]['total_samples'] * 100
        
        print(f"  {i} label(s): Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
        
        # Check if distributions are similar (within 5% difference)
        max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
        if max_diff > 5:
            print(f"    ⚠️  Warning: Large distribution difference ({max_diff:.1f}%)")
        else:
            print(f"    ✅ Good distribution consistency")
    
    print("\n✅ Dataset splits analysis completed!")

if __name__ == "__main__":
    main() 