"""
Analyze the semantic augmented dataset created from semantic pairs.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def load_augmented_dataset(file_path: str = 'data/semantic_augmented/semantic_augmented_dataset.csv') -> pd.DataFrame:
    """Load the augmented dataset."""
    print(f"Loading augmented dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} sentences")
    return df

def load_semantic_pairs(pairs_file: str = 'data/semantic_pairs.json') -> List[Dict]:
    """Load semantic pairs from JSON file."""
    print(f"Loading semantic pairs from {pairs_file}...")
    with open(pairs_file, 'r') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    print(f"Loaded {len(pairs)} semantic pairs")
    return pairs

def find_similar_sentences_from_pairs(target_sentence: str, pairs: List[Dict]) -> Dict[str, List[str]]:
    """Find sentences from each book that are similar to the target sentence."""
    similar_sentences = {
        'Anna Karenina': [],
        'Frankenstein': [],
        'The Adventures of Alice in Wonderland': [],
        'Wuthering Heights': []
    }
    
    for pair in pairs:
        if pair['sentence1'] == target_sentence:
            # Target sentence is sentence1, so sentence2 is from the other book
            other_book = pair['book2']
            other_sentence = pair['sentence2']
            if other_book in similar_sentences:
                similar_sentences[other_book].append(other_sentence)
        elif pair['sentence2'] == target_sentence:
            # Target sentence is sentence2, so sentence1 is from the other book
            other_book = pair['book1']
            other_sentence = pair['sentence1']
            if other_book in similar_sentences:
                similar_sentences[other_book].append(other_sentence)
    
    return similar_sentences

def analyze_dataset(df: pd.DataFrame, pairs: List[Dict]):
    """Analyze the augmented dataset and provide statistics."""
    print("\n" + "="*60)
    print("SEMANTIC AUGMENTED DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total sentences: {len(df)}")
    print(f"Unique sentences: {df['sentence'].nunique()}")
    
    # Book columns (excluding metadata columns)
    book_columns = [col for col in df.columns if col.startswith('book_')]
    print(f"Book columns: {book_columns}")
    
    # Original label distribution
    print(f"\n📚 ORIGINAL LABEL DISTRIBUTION:")
    label_counts = df['original_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"Label {label}: {count} sentences")
    
    # Original book distribution
    print(f"\n📖 ORIGINAL BOOK DISTRIBUTION:")
    book_counts = df['original_book'].value_counts()
    for book, count in book_counts.items():
        print(f"{book}: {count} sentences")
    
    # Multi-label statistics
    print(f"\n🏷️  MULTI-LABEL STATISTICS:")
    for book_col in book_columns:
        book_name = book_col.replace('book_', '').replace('_', ' ')
        count = df[book_col].sum()
        percentage = (count / len(df)) * 100
        print(f"{book_name}: {count} sentences ({percentage:.1f}%)")
    
    # Cross-book similarity analysis
    print(f"\n🔗 CROSS-BOOK SIMILARITY ANALYSIS:")
    for i, book1_col in enumerate(book_columns):
        for j, book2_col in enumerate(book_columns):
            if i < j:  # Avoid duplicates and self-comparison
                book1_name = book1_col.replace('book_', '').replace('_', ' ')
                book2_name = book2_col.replace('book_', '').replace('_', ' ')
                
                # Count sentences that are marked for both books
                both_marked = ((df[book1_col] == 1) & (df[book2_col] == 1)).sum()
                total_marked = ((df[book1_col] == 1) | (df[book2_col] == 1)).sum()
                
                if total_marked > 0:
                    similarity_percentage = (both_marked / total_marked) * 100
                    print(f"{book1_name} ↔ {book2_name}: {both_marked} shared sentences ({similarity_percentage:.1f}%)")
    
    # Label combinations analysis
    print(f"\n🎯 LABEL COMBINATION ANALYSIS:")
    # Create a combined label string for each row
    df['label_combination'] = df[book_columns].apply(
        lambda row: '+'.join([book_col for book_col in book_columns if row[book_col] == 1]), axis=1
    )
    
    combination_counts = df['label_combination'].value_counts()
    print("Top 10 label combinations:")
    for combo, count in combination_counts.head(10).items():
        if combo:  # Skip empty combinations
            books_in_combo = [col.replace('book_', '').replace('_', ' ') for col in combo.split('+')]
            print(f"  {books_in_combo}: {count} sentences")
    
    # Sentences with multiple labels
    df['num_labels'] = df[book_columns].sum(axis=1)
    print(f"\n📈 SENTENCES BY NUMBER OF LABELS:")
    label_count_dist = df['num_labels'].value_counts().sort_index()
    for num_labels, count in label_count_dist.items():
        print(f"{num_labels} label(s): {count} sentences")
    
    # Detailed similarity count analysis
    print(f"\n🔢 DETAILED SIMILARITY COUNT ANALYSIS:")
    print(f"Sentences with 1 book similarity: {label_count_dist.get(1, 0)} sentences")
    print(f"Sentences with 2 book similarities: {label_count_dist.get(2, 0)} sentences")
    print(f"Sentences with 3 book similarities: {label_count_dist.get(3, 0)} sentences")
    print(f"Sentences with 4 book similarities: {label_count_dist.get(4, 0)} sentences")
    
    # Percentage breakdown
    total_sentences = len(df)
    print(f"\n📊 PERCENTAGE BREAKDOWN:")
    for num_labels in range(1, 5):
        count = label_count_dist.get(num_labels, 0)
        percentage = (count / total_sentences) * 100
        print(f"{num_labels} book similarity: {count} sentences ({percentage:.1f}%)")
    
    # Sample sentences with multiple labels
    print(f"\n🔍 SAMPLE SENTENCES WITH MULTIPLE LABELS:")
    multi_label_sentences = df[df['num_labels'] > 1].head(10)
    for _, row in multi_label_sentences.iterrows():
        labels = [col.replace('book_', '').replace('_', ' ') for col in book_columns if row[col] == 1]
        print(f"  '{row['sentence'][:50]}...' → {labels}")
    
    # Detailed analysis of 4-book similarities
    print(f"\n🎯 DETAILED ANALYSIS OF 4-BOOK SIMILARITIES:")
    four_label_sentences = df[df['num_labels'] == 4]
    print(f"Found {len(four_label_sentences)} sentences with 4-book similarities")
    
    if len(four_label_sentences) > 0:
        print(f"\n📖 EXAMPLES OF 4-BOOK SIMILARITIES:")
        for i, (_, row) in enumerate(four_label_sentences.head(5).iterrows()):
            print(f"\n  Example {i+1}: '{row['sentence'][:100]}...'")
            print(f"    Original book: {row['original_book']}")
            print(f"    Similar to all 4 books: {[col.replace('book_', '').replace('_', ' ') for col in book_columns if row[col] == 1]}")
            
            # Find similar sentences from each book
            similar_sentences = find_similar_sentences_from_pairs(row['sentence'], pairs)
            if similar_sentences:
                print(f"    Contributing sentences from pairs:")
                for book, sentences in similar_sentences.items():
                    if sentences:
                        print(f"      {book}: {sentences[:3]}")  # Show first 3 sentences
            print()
    
    # Analysis of 3-book similarities
    print(f"\n📚 ANALYSIS OF 3-BOOK SIMILARITIES:")
    three_label_sentences = df[df['num_labels'] == 3]
    print(f"Found {len(three_label_sentences)} sentences with 3-book similarities")
    
    if len(three_label_sentences) > 0:
        print(f"\n📖 EXAMPLES OF 3-BOOK SIMILARITIES:")
        for i, (_, row) in enumerate(three_label_sentences.head(3).iterrows()):
            print(f"\n  Example {i+1}: '{row['sentence'][:100]}...'")
            print(f"    Original book: {row['original_book']}")
            similar_books = [col.replace('book_', '').replace('_', ' ') for col in book_columns if row[col] == 1]
            print(f"    Similar to: {similar_books}")
            
            # Find similar sentences from each book
            similar_sentences = find_similar_sentences_from_pairs(row['sentence'], pairs)
            if similar_sentences:
                print(f"    Contributing sentences from pairs:")
                for book, sentences in similar_sentences.items():
                    if sentences:
                        print(f"      {book}: {sentences[:2]}")  # Show first 2 sentences
            print()
    
    return df

def create_visualizations(df: pd.DataFrame):
    """Create visualizations for the augmented dataset."""
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original book distribution
    book_counts = df['original_book'].value_counts()
    axes[0, 0].bar(range(len(book_counts)), book_counts.values)
    axes[0, 0].set_title('Original Book Distribution')
    axes[0, 0].set_ylabel('Number of Sentences')
    axes[0, 0].set_xticks(range(len(book_counts)))
    axes[0, 0].set_xticklabels(book_counts.index, rotation=45, ha='right')
    
    # 2. Multi-label distribution
    book_columns = [col for col in df.columns if col.startswith('book_')]
    multi_label_counts = df[book_columns].sum()
    axes[0, 1].bar(range(len(multi_label_counts)), multi_label_counts.values)
    axes[0, 1].set_title('Multi-Label Distribution')
    axes[0, 1].set_ylabel('Number of Sentences')
    axes[0, 1].set_xticks(range(len(multi_label_counts)))
    axes[0, 1].set_xticklabels([col.replace('book_', '').replace('_', ' ') for col in book_columns], rotation=45, ha='right')
    
    # 3. Number of labels per sentence
    label_count_dist = df['num_labels'].value_counts().sort_index()
    axes[1, 0].bar(label_count_dist.index, label_count_dist.values)
    axes[1, 0].set_title('Sentences by Number of Labels')
    axes[1, 0].set_xlabel('Number of Labels')
    axes[1, 0].set_ylabel('Number of Sentences')
    
    # 4. Cross-book similarity heatmap
    similarity_matrix = np.zeros((len(book_columns), len(book_columns)))
    for i, book1_col in enumerate(book_columns):
        for j, book2_col in enumerate(book_columns):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                both_marked = ((df[book1_col] == 1) & (df[book2_col] == 1)).sum()
                total_marked = ((df[book1_col] == 1) | (df[book2_col] == 1)).sum()
                if total_marked > 0:
                    similarity_matrix[i, j] = both_marked / total_marked
    
    book_names = [col.replace('book_', '').replace('_', ' ') for col in book_columns]
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', 
                xticklabels=book_names, yticklabels=book_names, ax=axes[1, 1])
    axes[1, 1].set_title('Cross-Book Similarity Matrix')
    
    plt.tight_layout()
    plt.savefig('data/semantic_augmented/semantic_augmented_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Visualizations saved to data/semantic_augmented/semantic_augmented_analysis.png")
    
    plt.show()

def save_analysis_report(df: pd.DataFrame, output_path: str = 'data/semantic_augmented/semantic_augmented_analysis.json'):
    """Save detailed analysis report."""
    book_columns = [col for col in df.columns if col.startswith('book_')]
    
    # Calculate statistics
    stats = {
        'total_sentences': len(df),
        'unique_sentences': df['sentence'].nunique(),
        'original_label_distribution': df['original_label'].value_counts().to_dict(),
        'original_book_distribution': df['original_book'].value_counts().to_dict(),
        'multi_label_distribution': {col.replace('book_', '').replace('_', ' '): int(df[col].sum()) 
                                   for col in book_columns},
        'label_count_distribution': df['num_labels'].value_counts().sort_index().to_dict(),
        'cross_book_similarities': {}
    }
    
    # Calculate cross-book similarities
    for i, book1_col in enumerate(book_columns):
        for j, book2_col in enumerate(book_columns):
            if i < j:
                book1_name = book1_col.replace('book_', '').replace('_', ' ')
                book2_name = book2_col.replace('book_', '').replace('_', ' ')
                
                both_marked = ((df[book1_col] == 1) & (df[book2_col] == 1)).sum()
                total_marked = ((df[book1_col] == 1) | (df[book2_col] == 1)).sum()
                
                if total_marked > 0:
                    similarity_percentage = (both_marked / total_marked) * 100
                    stats['cross_book_similarities'][f"{book1_name}_vs_{book2_name}"] = {
                        'shared_sentences': int(both_marked),
                        'total_marked': int(total_marked),
                        'similarity_percentage': round(similarity_percentage, 2)
                    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📄 Analysis report saved to {output_path}")

def main():
    """Main function to analyze the augmented dataset."""
    print("🔍 Analyzing semantic augmented dataset...")
    
    # Load dataset and pairs
    df = load_augmented_dataset()
    pairs = load_semantic_pairs()
    
    # Analyze dataset
    df = analyze_dataset(df, pairs)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save analysis report
    save_analysis_report(df)
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main() 