"""
Create augmented dataset from semantic pairs.
Each sentence from a pair becomes a data point with multi-label annotations.
"""

import json
import pandas as pd
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from typing import Dict, List, Tuple
import logging
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_semantic_pairs(pairs_file: str = 'data/semantic_pairs.json') -> List[Dict]:
    """Load semantic pairs from JSON file."""
    logger.info(f"Loading semantic pairs from {pairs_file}...")
    with open(pairs_file, 'r') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    logger.info(f"Loaded {len(pairs)} semantic pairs")
    return pairs

def load_deduplicated_corpus(corpus_path: str = 'data/corpus_deduplicated.json') -> Dict:
    """Load the deduplicated corpus."""
    logger.info(f"Loading deduplicated corpus from {corpus_path}...")
    with open(corpus_path, 'r') as f:
        corpus_data = json.load(f)
    
    total_sentences = sum(len(book_data['sentences']) for book_data in corpus_data.values())
    logger.info(f"Loaded {total_sentences} total sentences from deduplicated corpus")
    logger.info(f"Books: {list(corpus_data.keys())}")
    
    return corpus_data

def create_augmented_dataset(pairs: List[Dict], corpus_data: Dict, 
                           min_similarity: float = 0.7) -> pd.DataFrame:
    """
    Create augmented dataset from semantic pairs.
    
    Args:
        pairs: List of semantic pairs
        corpus_data: Deduplicated corpus data
        min_similarity: Minimum similarity threshold
    
    Returns:
        DataFrame with columns: sentence, original_label, book_columns (0/1)
    """
    logger.info("Creating augmented dataset from semantic pairs...")
    
    # Get unique book names from corpus
    books = sorted(list(corpus_data.keys()))
    logger.info(f"Books in corpus: {books}")
    
    # Create book to index mapping
    book_to_idx = {book: idx for idx, book in enumerate(books)}
    
    # Initialize augmented data with ALL sentences from corpus
    augmented_data = []
    
    # First, add all sentences from the corpus with their original labels
    for book_name, book_data in corpus_data.items():
        book_label = list(corpus_data.keys()).index(book_name)  # 0, 1, 2, 3
        
        for sentence in book_data['sentences']:
            # Create row for augmented dataset
            row = {
                'sentence': sentence,
                'original_label': book_label,
                'original_book': book_name
            }
            
            # Initialize all book columns to 0
            for book in books:
                row[f'book_{book.replace(" ", "_").replace("'", "")}'] = 0
            
            # Set the original book column to 1
            row[f'book_{book_name.replace(" ", "_").replace("'", "")}'] = 1
            
            augmented_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(augmented_data)
    logger.info(f"Initial dataset with all corpus sentences: {len(df)} sentences")
    
    # Now process semantic pairs to add additional book labels
    pairs_processed = 0
    for pair in pairs:
        if pair['similarity'] < min_similarity:
            continue
            
        # Get sentences and their book information
        sentence1 = pair['sentence1']
        sentence2 = pair['sentence2']
        book1 = pair['book1']
        book2 = pair['book2']
        
        # Find and update both sentences in the dataset
        for sentence, paired_book in [(sentence1, book2), (sentence2, book1)]:
            # Find the sentence in the dataset
            mask = df['sentence'] == sentence
            if mask.any():
                # Add the paired book label
                book_col = f'book_{paired_book.replace(" ", "_").replace("'", "")}'
                if book_col in df.columns:
                    df.loc[mask, book_col] = 1
                    pairs_processed += 1
            else:
                logger.warning(f"Could not find sentence in corpus: {sentence[:50]}...")
    
    logger.info(f"Processed {pairs_processed} sentence updates from semantic pairs")
    logger.info(f"Final augmented dataset: {len(df)} sentences")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Log statistics
    book_columns = [col for col in df.columns if col.startswith('book_')]
    for book_col in book_columns:
        book_name = book_col.replace('book_', '').replace('_', ' ')
        count = df[book_col].sum()
        logger.info(f"{book_name}: {count} sentences marked")
    
    return df

def save_augmented_dataset(df: pd.DataFrame, output_path: str = 'data/semantic_augmented/semantic_augmented_dataset.csv'):
    """Save the augmented dataset."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved augmented dataset to {output_path}")
    
    # Also save as JSON for easier inspection
    json_path = output_path.replace('.csv', '.json')
    df.to_json(json_path, orient='records', indent=2)
    logger.info(f"Saved augmented dataset to {json_path}")

def create_dataset_splits(df: pd.DataFrame, config_path: str = "configs/config.yaml") -> Tuple[DatasetDict, Dict]:
    """Create train/validation/test splits from the augmented dataset."""
    logger.info("Creating dataset splits...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create label mapping for multi-label classification
    book_columns = [col for col in df.columns if col.startswith('book_')]
    books = [col.replace('book_', '').replace('_', ' ') for col in book_columns]
    
    # Create multi-label format
    labels = df[book_columns].values.tolist()
    
    # Create dataset with multi-label format
    dataset_df = pd.DataFrame({
        'sentence': df['sentence'],
        'labels': labels,
        'original_label': df['original_label'],
        'original_book': df['original_book']
    })
    
    # Split data
    train_df, temp_df = train_test_split(
        dataset_df, 
        test_size=1-config['data']['train_split'],
        random_state=config['data']['random_seed']
    )
    
    val_size = config['data']['val_split'] / (config['data']['val_split'] + config['data']['test_split'])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1-val_size,
        random_state=config['data']['random_seed']
    )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Calculate label distribution
    num_labels_per_sentence = df[book_columns].sum(axis=1)
    label_distribution = {
        '1_label': len(num_labels_per_sentence[num_labels_per_sentence == 1]),
        '2_labels': len(num_labels_per_sentence[num_labels_per_sentence == 2]),
        '3_labels': len(num_labels_per_sentence[num_labels_per_sentence == 3]),
        '4_labels': len(num_labels_per_sentence[num_labels_per_sentence == 4])
    }
    
    # Create metadata
    metadata = {
        'books': books,
        'num_classes': len(books),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(dataset_df),
        'multi_label': True,
        'dataset_type': 'semantic_augmented',
        'book_columns': book_columns,
        'label_distribution': label_distribution
    }
    
    return dataset_dict, metadata

def main():
    """Main function to create augmented dataset."""
    logger.info("Starting semantic augmented dataset creation...")
    
    # Load data
    pairs = load_semantic_pairs()
    corpus_data = load_deduplicated_corpus()
    
    # Create augmented dataset
    augmented_df = create_augmented_dataset(pairs, corpus_data, min_similarity=0.7)
    
    # Save augmented dataset
    save_augmented_dataset(augmented_df)
    
    # Create dataset splits
    dataset_dict, metadata = create_dataset_splits(augmented_df)
    
    # Save dataset splits
    dataset_path = Path("data/dataset")
    dataset_dict.save_to_disk(str(dataset_path))
    
    # Save metadata
    metadata_path = Path("data/metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✅ Semantic augmented dataset creation completed!")
    logger.info(f"📁 Dataset saved to: {dataset_path}")
    logger.info(f"📊 Metadata saved to: {metadata_path}")
    
    # Print summary statistics
    print("\n=== AUGMENTED DATASET SUMMARY ===")
    print(f"Total sentences: {len(augmented_df)}")
    print(f"Dataset splits: Train={metadata['train_size']}, Val={metadata['val_size']}, Test={metadata['test_size']}")
    print(f"Multi-label classes: {metadata['num_classes']}")
    print(f"Label distribution: {metadata['label_distribution']}")
    
    # Show sample data
    print("\n=== SAMPLE DATA ===")
    print(augmented_df.head())

if __name__ == "__main__":
    main() 