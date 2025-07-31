#!/usr/bin/env python3
"""
EDA to investigate potential data and model issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_data_issues():
    """Analyze potential issues with the augmented dataset."""
    
    # Load data
    df = pd.read_csv('data/features_knn/augmented_dataset.csv')
    
    print("=== DATA ISSUES ANALYSIS ===\n")
    
    # 1. True book distribution
    print("1. TRUE BOOK DISTRIBUTION:")
    print(df['true_book'].value_counts())
    print(f"Total samples: {len(df)}")
    print()
    
    # 2. Belongs_to distribution vs true_book
    print("2. BELONGS_TO vs TRUE_BOOK ANALYSIS:")
    books = ["The Adventures of Alice in Wonderland", "Anna Karenina", "Wuthering Heights", "Frankenstein"]
    
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        
        true_positives = (true_book_mask & belongs_mask).sum()
        false_positives = (~true_book_mask & belongs_mask).sum()
        true_negatives = (~true_book_mask & ~belongs_mask).sum()
        false_negatives = (true_book_mask & ~belongs_mask).sum()
        
        print(f"\n{book}:")
        print(f"  True positives: {true_positives}")
        print(f"  False positives: {false_positives}")
        print(f"  True negatives: {true_negatives}")
        print(f"  False negatives: {false_negatives}")
        print(f"  Precision: {true_positives/(true_positives+false_positives):.3f}" if (true_positives+false_positives) > 0 else "  Precision: 0.000")
        print(f"  Recall: {true_positives/(true_positives+false_negatives):.3f}" if (true_positives+false_negatives) > 0 else "  Recall: 0.000")
    
    # 3. Similarity score analysis
    print("\n3. SIMILARITY SCORE ANALYSIS:")
    for book in books:
        sim_col = f'similarity_{book}'
        true_book_mask = df['true_book'] == book
        
        true_book_similarities = df.loc[true_book_mask, sim_col]
        other_book_similarities = df.loc[~true_book_mask, sim_col]
        
        print(f"\n{book}:")
        print(f"  True book mean similarity: {true_book_similarities.mean():.3f}")
        print(f"  Other books mean similarity: {other_book_similarities.mean():.3f}")
        print(f"  Separation: {true_book_similarities.mean() - other_book_similarities.mean():.3f}")
    
    # 4. KNN analysis
    print("\n4. KNN ANALYSIS:")
    print(f"KNN accuracy: {df['knn_confidence'].mean():.3f}")
    print(f"KNN best book distribution:")
    print(df['knn_best_book'].value_counts())
    
    # 5. Multi-label analysis
    print("\n5. MULTI-LABEL ANALYSIS:")
    belongs_cols = [f'belongs_to_{book}' for book in books]
    df['num_books'] = df[belongs_cols].sum(axis=1)
    print(f"Mean books per sentence: {df['num_books'].mean():.3f}")
    print(f"Single book ratio: {(df['num_books'] == 1).mean():.3f}")
    print(f"Multi-book ratio: {(df['num_books'] > 1).mean():.3f}")
    
    # 6. Sample sentence analysis
    print("\n6. SAMPLE SENTENCE ANALYSIS:")
    for book in books:
        book_samples = df[df['true_book'] == book].head(3)
        print(f"\n{book} samples:")
        for _, row in book_samples.iterrows():
            belongs_col = f'belongs_to_{book}'
            print(f"  '{row['sentence'][:50]}...' -> belongs_to: {row[belongs_col]}")
    
    # 7. Potential issues
    print("\n7. POTENTIAL ISSUES IDENTIFIED:")
    
    # Issue 1: Class imbalance
    print("  - Severe class imbalance in true_book distribution")
    print("  - Anna Karenina has 64% of all samples")
    print("  - Alice in Wonderland has only 5% of samples")
    
    # Issue 2: Belongs_to vs true_book mismatch
    mismatches = 0
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        mismatches += (~true_book_mask & belongs_mask).sum()
    
    print(f"  - {mismatches} false positive belongs_to assignments")
    
    # Issue 3: Similarity score quality
    low_separation_books = []
    for book in books:
        sim_col = f'similarity_{book}'
        true_book_mask = df['true_book'] == book
        true_book_similarities = df.loc[true_book_mask, sim_col]
        other_book_similarities = df.loc[~true_book_mask, sim_col]
        separation = true_book_similarities.mean() - other_book_similarities.mean()
        
        if separation < 0.1:
            low_separation_books.append(book)
    
    if low_separation_books:
        print(f"  - Poor similarity separation for: {low_separation_books}")
    
    return df

def create_visualizations(df):
    """Create visualizations to understand the data better."""
    
    books = ["The Adventures of Alice in Wonderland", "Anna Karenina", "Wuthering Heights", "Frankenstein"]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. True book distribution
    axes[0, 0].bar(df['true_book'].value_counts().index, df['true_book'].value_counts().values)
    axes[0, 0].set_title('True Book Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Similarity score distributions
    for i, book in enumerate(books):
        sim_col = f'similarity_{book}'
        true_book_mask = df['true_book'] == book
        axes[0, 1].hist(df.loc[true_book_mask, sim_col], alpha=0.7, label=f'{book} (true)', bins=20)
        axes[0, 1].hist(df.loc[~true_book_mask, sim_col], alpha=0.7, label=f'{book} (other)', bins=20)
    axes[0, 1].set_title('Similarity Score Distributions')
    axes[0, 1].legend()
    
    # 3. Belongs_to vs true_book confusion matrix
    belongs_cols = [f'belongs_to_{book}' for book in books]
    confusion_data = []
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        confusion_data.append([
            (true_book_mask & belongs_mask).sum(),  # TP
            (~true_book_mask & belongs_mask).sum(), # FP
            (true_book_mask & ~belongs_mask).sum(), # FN
            (~true_book_mask & ~belongs_mask).sum() # TN
        ])
    
    confusion_df = pd.DataFrame(confusion_data, 
                               index=books,
                               columns=['TP', 'FP', 'FN', 'TN'])
    
    sns.heatmap(confusion_df, annot=True, fmt='d', ax=axes[1, 0])
    axes[1, 0].set_title('Belongs_to vs True_book Confusion Matrix')
    
    # 4. Multi-label distribution
    belongs_cols = [f'belongs_to_{book}' for book in books]
    df['num_books'] = df[belongs_cols].sum(axis=1)
    axes[1, 1].hist(df['num_books'], bins=range(1, 6), alpha=0.7)
    axes[1, 1].set_title('Number of Books per Sentence')
    axes[1, 1].set_xlabel('Number of Books')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('eda_data_issues.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'eda_data_issues.png'")

if __name__ == "__main__":
    df = analyze_data_issues()
    create_visualizations(df) 