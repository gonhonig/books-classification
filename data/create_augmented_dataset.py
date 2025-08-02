"""
Create augmented dataset from semantic pairs.
Each sentence from a pair becomes a data point with multi-label annotations.
Includes balanced dataset creation, comprehensive statistics, and aligned embeddings.
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
from sentence_transformers import SentenceTransformer
import hashlib

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

def create_balanced_dataset(df: pd.DataFrame, target_samples_per_class: int = 5000) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a balanced dataset prioritizing multi-label sentences.
    
    Args:
        df: Augmented dataset DataFrame
        target_samples_per_class: Target number of samples per class
    
    Returns:
        Balanced DataFrame and statistics
    """
    logger.info("Creating balanced dataset prioritizing multi-label sentences...")
    
    # Get book columns
    book_columns = [col for col in df.columns if col.startswith('book_')]
    books = [col.replace('book_', '').replace('_', ' ') for col in book_columns]
    
    # Create book labels array for easier processing
    book_labels = df[book_columns].values
    
    # Identify multi-label sentences
    multi_label_indices = []
    for i in range(len(df)):
        if np.sum(book_labels[i]) > 1:
            multi_label_indices.append(i)
    
    logger.info(f"Found {len(multi_label_indices)} multi-label sentences")
    
    # Include ALL multi-label sentences
    selected_indices = multi_label_indices.copy()
    logger.info(f"Included {len(selected_indices)} multi-label sentences")
    
    # For each book, add single-label positive samples to reach target
    for book_col in book_columns:
        # Get single-label positive samples for this book
        positive_indices = np.where(book_labels[:, book_columns.index(book_col)] == 1)[0]
        single_label_positive = [i for i in positive_indices if i not in multi_label_indices]
        
        # Calculate how many more positive samples we need
        current_positive_count = sum(1 for i in selected_indices if book_labels[i, book_columns.index(book_col)] == 1)
        additional_needed = target_samples_per_class - current_positive_count
        
        if additional_needed > 0 and len(single_label_positive) > 0:
            # Sample additional single-label positive samples
            n_to_sample = min(additional_needed, len(single_label_positive))
            np.random.seed(42)  # For reproducibility
            selected_single_label = np.random.choice(single_label_positive, n_to_sample, replace=False)
            selected_indices.extend(selected_single_label)
            logger.info(f"Added {n_to_sample} single-label positive samples for {book_col}")
    
    # Add negative samples to balance the dataset
    # We want equal numbers of positive and negative samples for each book
    for book_col in book_columns:
        positive_count = sum(1 for i in selected_indices if book_labels[i, book_columns.index(book_col)] == 1)
        negative_count = sum(1 for i in selected_indices if book_labels[i, book_columns.index(book_col)] == 0)
        
        if negative_count < positive_count:
            # Need more negative samples
            additional_negative_needed = positive_count - negative_count
            negative_indices = np.where(book_labels[:, book_columns.index(book_col)] == 0)[0]
            available_negative = [i for i in negative_indices if i not in selected_indices]
            
            if len(available_negative) > 0:
                n_to_sample = min(additional_negative_needed, len(available_negative))
                np.random.seed(42)
                selected_negative = np.random.choice(available_negative, n_to_sample, replace=False)
                selected_indices.extend(selected_negative)
                logger.info(f"Added {n_to_sample} negative samples for {book_col}")
    
    # Remove duplicates and shuffle
    selected_indices = list(set(selected_indices))
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    # Create the balanced dataset
    balanced_df = df.iloc[selected_indices].reset_index(drop=True)
    
    # Calculate statistics
    stats = {
        'total_samples': len(balanced_df),
        'multi_label_samples': len(multi_label_indices),
        'single_label_samples': len(balanced_df) - len(multi_label_indices),
        'book_statistics': {}
    }
    
    for book_col in book_columns:
        book_name = book_col.replace('book_', '').replace('_', ' ')
        positive_count = balanced_df[book_col].sum()
        negative_count = len(balanced_df) - positive_count
        stats['book_statistics'][book_name] = {
            'positive_samples': int(positive_count),
            'negative_samples': int(negative_count),
            'total_samples': len(balanced_df)
        }
    
    logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
    for book_name, book_stats in stats['book_statistics'].items():
        logger.info(f"{book_name}: {book_stats['positive_samples']} positive, {book_stats['negative_samples']} negative")
    
    return balanced_df, stats

def create_dataset_splits(df: pd.DataFrame, config_path: str = "configs/config.yaml") -> Tuple[DatasetDict, Dict]:
    """Create train/validation/test splits from the balanced dataset."""
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
    
    # Calculate comprehensive statistics
    num_labels_per_sentence = df[book_columns].sum(axis=1)
    label_distribution = {
        '1_label': len(num_labels_per_sentence[num_labels_per_sentence == 1]),
        '2_labels': len(num_labels_per_sentence[num_labels_per_sentence == 2]),
        '3_labels': len(num_labels_per_sentence[num_labels_per_sentence == 3]),
        '4_labels': len(num_labels_per_sentence[num_labels_per_sentence == 4])
    }
    
    # Calculate per-split statistics
    split_statistics = {}
    for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        split_labels = np.array(split_df['labels'].tolist())
        split_num_labels = np.sum(split_labels, axis=1)
        
        split_statistics[split_name] = {
            'total_samples': len(split_df),
            'label_distribution': {
                '1_label': int(np.sum(split_num_labels == 1)),
                '2_labels': int(np.sum(split_num_labels == 2)),
                '3_labels': int(np.sum(split_num_labels == 3)),
                '4_labels': int(np.sum(split_num_labels == 4))
            },
            'book_distribution': {}
        }
        
        # Calculate book distribution for this split
        for i, book_col in enumerate(book_columns):
            book_name = books[i]
            positive_count = int(np.sum(split_labels[:, i] == 1))
            negative_count = len(split_df) - positive_count
            split_statistics[split_name]['book_distribution'][book_name] = {
                'positive_samples': positive_count,
                'negative_samples': negative_count
            }
    
    # Create comprehensive metadata
    metadata = {
        'books': books,
        'num_classes': len(books),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(dataset_df),
        'multi_label': True,
        'dataset_type': 'semantic_augmented_balanced',
        'book_columns': book_columns,
        'label_distribution': label_distribution,
        'split_statistics': split_statistics,
        'balanced_dataset_stats': {
            'target_samples_per_class': 5000,
            'multi_label_priority': True,
            'balanced_positive_negative': True
        }
    }
    
    return dataset_dict, metadata

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

def save_comprehensive_metadata(metadata: Dict, output_path: str = 'data/semantic_augmented/dataset_metadata.json'):
    """Save comprehensive dataset metadata."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved comprehensive metadata to {output_path}")
    
    # Also create a human-readable summary
    summary_path = output_path.replace('.json', '_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# Semantic Augmented Dataset - Comprehensive Statistics\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- **Total Samples**: {metadata['total_size']}\n")
        f.write(f"- **Number of Classes**: {metadata['num_classes']}\n")
        f.write(f"- **Dataset Type**: {metadata['dataset_type']}\n")
        f.write(f"- **Multi-label**: {metadata['multi_label']}\n\n")
        
        f.write("## Split Distribution\n")
        f.write(f"- **Train**: {metadata['train_size']} samples\n")
        f.write(f"- **Validation**: {metadata['val_size']} samples\n")
        f.write(f"- **Test**: {metadata['test_size']} samples\n\n")
        
        f.write("## Overall Label Distribution\n")
        for label_count, count in metadata['label_distribution'].items():
            f.write(f"- **{label_count}**: {count} samples\n")
        f.write("\n")
        
        f.write("## Per-Split Statistics\n")
        for split_name, split_stats in metadata['split_statistics'].items():
            f.write(f"### {split_name.title()} Split\n")
            f.write(f"- **Total Samples**: {split_stats['total_samples']}\n")
            f.write("- **Label Distribution**:\n")
            for label_count, count in split_stats['label_distribution'].items():
                f.write(f"  - {label_count}: {count} samples\n")
            f.write("- **Book Distribution**:\n")
            for book_name, book_stats in split_stats['book_distribution'].items():
                f.write(f"  - {book_name}: {book_stats['positive_samples']} positive, {book_stats['negative_samples']} negative\n")
            f.write("\n")
        
        f.write("## Balanced Dataset Configuration\n")
        for key, value in metadata['balanced_dataset_stats'].items():
            f.write(f"- **{key}**: {value}\n")
    
    logger.info(f"Saved dataset summary to {summary_path}")

def create_aligned_embeddings(dataset_dict: DatasetDict, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Create embeddings that are perfectly aligned with the dataset splits."""
    logger.info("Creating aligned embeddings for dataset splits...")
    
    # Load the sentence transformer model
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Collect all sentences in the correct order
    all_sentences = []
    for split_name in ['train', 'validation', 'test']:
        all_sentences.extend(dataset_dict[split_name]['sentence'])
    
    logger.info(f"Processing {len(all_sentences)} sentences...")
    
    # Create embeddings
    embeddings = model.encode(all_sentences, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Created embeddings with shape: {embeddings.shape}")
    
    # Create hash for the dataset
    config = {
        'model_name': model_name,
        'dataset_size': len(all_sentences),
        'embedding_dim': embeddings.shape[1],
        'splits': {
            'train': len(dataset_dict['train']),
            'validation': len(dataset_dict['validation']),
            'test': len(dataset_dict['test'])
        }
    }
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # Save embeddings
    output_path = f'data/embeddings_cache_aligned_{config_hash}.npz'
    np.savez_compressed(output_path, embeddings=embeddings)
    logger.info(f"Saved aligned embeddings to {output_path}")
    
    # Save config hash
    hash_path = f'data/embeddings_cache_aligned_{config_hash}.txt'
    with open(hash_path, 'w') as f:
        f.write(config_hash)
    
    logger.info(f"Saved config hash to {hash_path}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Config hash: {config_hash}")
    
    return output_path, config_hash

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
    
    # Create balanced dataset
    balanced_df, balanced_stats = create_balanced_dataset(augmented_df, target_samples_per_class=5000)
    
    # Save balanced dataset
    balanced_output_path = 'data/semantic_augmented/semantic_augmented_balanced_dataset.csv'
    balanced_df.to_csv(balanced_output_path, index=False)
    logger.info(f"Saved balanced dataset to {balanced_output_path}")
    
    # Create dataset splits
    dataset_dict, metadata = create_dataset_splits(balanced_df)
    
    # Create aligned embeddings for the dataset splits
    embeddings_path, embeddings_hash = create_aligned_embeddings(dataset_dict)
    
    # Add balanced dataset statistics to metadata
    metadata['balanced_dataset_stats'].update(balanced_stats)
    
    # Add embedding information to metadata
    metadata['embeddings'] = {
        'path': embeddings_path,
        'hash': embeddings_hash,
        'aligned': True
    }
    
    # Save dataset splits
    dataset_path = Path("data/dataset")
    dataset_dict.save_to_disk(str(dataset_path))
    
    # Save comprehensive metadata
    save_comprehensive_metadata(metadata)
    
    logger.info("✅ Semantic augmented dataset creation completed!")
    logger.info(f"📁 Dataset saved to: {dataset_path}")
    logger.info(f"📁 Aligned embeddings saved to: {embeddings_path}")
    logger.info(f"📊 Comprehensive metadata saved")
    
    # Print summary statistics
    print("\n=== AUGMENTED DATASET SUMMARY ===")
    print(f"Original dataset: {len(augmented_df)} sentences")
    print(f"Balanced dataset: {len(balanced_df)} sentences")
    print(f"Dataset splits: Train={metadata['train_size']}, Val={metadata['val_size']}, Test={metadata['test_size']}")
    print(f"Multi-label classes: {metadata['num_classes']}")
    print(f"Label distribution: {metadata['label_distribution']}")
    print(f"Multi-label samples: {balanced_stats['multi_label_samples']}")
    print(f"Single-label samples: {balanced_stats['single_label_samples']}")
    print(f"Aligned embeddings: {embeddings_path}")
    print(f"Embeddings hash: {embeddings_hash}")
    
    # Show sample data
    print("\n=== SAMPLE DATA ===")
    print(balanced_df.head())

if __name__ == "__main__":
    main() 