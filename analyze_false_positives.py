#!/usr/bin/env python3
"""
Analyze false positives in the belongs_to features to understand if they're realistic.
"""

import pandas as pd
import numpy as np

def analyze_false_positives():
    """Analyze false positives in belongs_to features."""
    
    df = pd.read_csv('data/features_knn/augmented_dataset.csv')
    books = ['The Adventures of Alice in Wonderland', 'Anna Karenina', 'Wuthering Heights', 'Frankenstein']
    
    print("=== FALSE POSITIVE ANALYSIS ===\n")
    
    total_fp = 0
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        fp = (~true_book_mask & belongs_mask).sum()
        total_fp += fp
        
        print(f"{book}:")
        print(f"  False positives: {fp} ({fp/len(df)*100:.1f}% of all sentences)")
        print(f"  True positives: {(true_book_mask & belongs_mask).sum()}")
        print()
    
    print(f"Total false positives: {total_fp} ({total_fp/len(df)*100:.1f}% of all sentences)")
    
    # Analyze some examples of false positives
    print("\n=== FALSE POSITIVE EXAMPLES ===")
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        fp_mask = (~true_book_mask & belongs_mask)
        
        if fp_mask.sum() > 0:
            fp_examples = df[fp_mask].head(3)
            print(f"\n{book} false positive examples:")
            for _, row in fp_examples.iterrows():
                print(f"  True book: {row['true_book']}")
                print(f"  Sentence: '{row['sentence'][:80]}...'")
                print(f"  Similarity: {row[f'similarity_{book}']:.3f}")
                print()
    
    # Check if false positives are semantically reasonable
    print("=== SEMANTIC REASONABLENESS ===")
    for book in books:
        belongs_col = f'belongs_to_{book}'
        true_book_mask = df['true_book'] == book
        belongs_mask = df[belongs_col] == 1
        fp_mask = (~true_book_mask & belongs_mask)
        
        if fp_mask.sum() > 0:
            fp_similarities = df.loc[fp_mask, f'similarity_{book}']
            print(f"\n{book} false positive similarity stats:")
            print(f"  Mean similarity: {fp_similarities.mean():.3f}")
            print(f"  Min similarity: {fp_similarities.min():.3f}")
            print(f"  Max similarity: {fp_similarities.max():.3f}")
            print(f"  High similarity (>0.5): {(fp_similarities > 0.5).sum()} ({fp_similarities.mean()*100:.1f}%)")

if __name__ == "__main__":
    analyze_false_positives() 