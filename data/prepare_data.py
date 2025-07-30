"""
Data preparation for English book classification project.
Step 1: Clean and preprocess English book descriptions.
"""

import os
import json
import logging
import argparse
import hashlib
import sys
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataPreparator:
    """Data preparation for English book classification."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data preparator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        np.random.seed(self.config['data']['random_seed'])
        
    def _get_config_hash(self) -> str:
        """Generate a hash of the data config for change detection."""
        config_str = json.dumps(self.config['data'], sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
        
    def needs_recreation(self, force_recreate: bool = False) -> bool:
        """Check if data needs to be recreated."""
        if force_recreate:
            self.logger.info("Force recreation requested")
            return True
            
        # Check if required files exist
        required_files = [
            self.data_dir / "raw_corpus.json",
            self.data_dir / "processed_dataset",
            self.data_dir / "metadata.json"
        ]
        
        if not all(f.exists() for f in required_files):
            self.logger.info("Some required files are missing")
            return True
            
        # Check if config has changed
        config_hash_file = self.data_dir / "config_hash.txt"
        current_hash = self._get_config_hash()
        
        if not config_hash_file.exists():
            return True
            
        with open(config_hash_file, 'r') as f:
            stored_hash = f.read().strip()
            
        if current_hash != stored_hash:
            self.logger.info("Configuration has changed")
            return True
            
        self.logger.info("All files exist and config unchanged - using existing files")
        return False
        
    def download_corpus(self) -> Dict:
        """Download and filter the book corpus."""
        self.logger.info("Loading English books dataset...")
        
        try:
            from datasets import load_dataset
            
            # Load the dataset
            train_ds = load_dataset("IsmaelMousa/books", split="train")
            val_ds = load_dataset("IsmaelMousa/books", split="validation")
            dataset = list(train_ds) + list(val_ds)

            # Filter for selected books
            selected_titles = self.config['data']['selected_books']
            max_sentences = self.config['data'].get('max_sentences_per_book')
            
            corpus_data = {}
            self.logger.info(f"Filtering for selected books: {selected_titles}")
            
            for row in dataset:
                title = row.get('title', '')
                if title in selected_titles:
                    if title not in corpus_data:
                        # Extract sentences from the EN field
                        text = row.get('EN', '')
                        sentences = sent_tokenize(text)
                        
                        # Apply sentence limit if specified
                        if max_sentences and len(sentences) > max_sentences:
                            sentences = sentences[:max_sentences]
                            self.logger.info(f"Limited '{title}' to {max_sentences} sentences")
                        
                        corpus_data[title] = {
                            "title": title,
                            "author": row.get('author', 'Unknown'),
                            "genre": row.get('category', 'Unknown'),
                            "sentences": sentences
                        }
                        self.logger.info(f"Collected {len(sentences)} sentences for '{title}'")
            
            # Save the corpus
            corpus_path = self.data_dir / "raw_corpus.json"
            with open(corpus_path, 'w') as f:
                json.dump(corpus_data, f, indent=2)
            
            self.logger.info(f"Corpus saved to {corpus_path}")
            self.logger.info(f"Total books collected: {len(corpus_data)}")
            
            return corpus_data
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """Preprocess sentences according to configuration."""
        processed = []
        preprocess_config = self.config['data']['preprocessing']
        
        for sentence in sentences:
            # Clean and normalize
            sentence = sentence.strip()
            
            # Apply preprocessing options
            if preprocess_config.get('normalize_text'):
                sentence = sentence.replace('\n', ' ').replace('\r', ' ')
                sentence = ' '.join(sentence.split())  # Normalize whitespace
            
            if preprocess_config.get('remove_special_chars'):
                import re
                sentence = re.sub(r'[^\w\s]', '', sentence)
            
            if preprocess_config.get('lowercase'):
                sentence = sentence.lower()
            
            # Filter by length
            if (len(sentence) >= self.config['data']['min_sentence_length'] and 
                len(sentence) <= self.config['data']['max_sentence_length']):
                processed.append(sentence)
                
        # Remove duplicates if configured
        if preprocess_config.get('remove_duplicates'):
            processed = list(dict.fromkeys(processed))  # Preserve order
                
        return processed
        
    def create_dataset(self) -> Tuple[DatasetDict, Dict]:
        """Create the final dataset for training."""
        self.logger.info("Creating dataset...")
        
        # Load corpus
        corpus_path = self.data_dir / "raw_corpus.json"
        if not corpus_path.exists():
            self.download_corpus()
            
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        
        # Prepare data
        all_sentences = []
        all_labels = []
        label_to_id = {}
        
        for book_id, book_data in corpus.items():
            if book_id not in label_to_id:
                label_to_id[book_id] = len(label_to_id)
                
            sentences = self.preprocess_sentences(book_data['sentences'])
            
            for sentence in sentences:
                all_sentences.append(sentence)
                all_labels.append(label_to_id[book_id])
        
        # Create DataFrame
        df = pd.DataFrame({
            'sentence': all_sentences,
            'label': all_labels,
            'book_id': [list(corpus.keys())[label] for label in all_labels]
        })
        
        # Split data
        train_df, temp_df = train_test_split(
            df, 
            test_size=1-self.config['data']['train_split'],
            stratify=df['label'],
            random_state=self.config['data']['random_seed']
        )
        
        val_size = self.config['data']['val_split'] / (self.config['data']['val_split'] + self.config['data']['test_split'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1-val_size,
            stratify=temp_df['label'],
            random_state=self.config['data']['random_seed']
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
        
        # Save metadata
        metadata = {
            'label_to_id': label_to_id,
            'id_to_label': {v: k for k, v in label_to_id.items()},
            'num_classes': len(label_to_id),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(df),
            'books': list(corpus.keys())
        }
        
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save processed dataset
        dataset_path = self.data_dir / "processed_dataset"
        dataset_dict.save_to_disk(str(dataset_path))
        
        self.logger.info(f"Dataset created and saved to {dataset_path}")
        self.logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return dataset_dict, metadata
        
    def save_config_hash(self):
        """Save current config hash."""
        config_hash_file = self.data_dir / "config_hash.txt"
        current_hash = self._get_config_hash()
        
        with open(config_hash_file, 'w') as f:
            f.write(current_hash)

def prepare_data(force_recreate=False, config_path="configs/config.yaml"):
    """Prepare the dataset for English book classification."""
    preparator = DataPreparator(config_path=config_path)
    
    # Check if recreation is needed
    if preparator.needs_recreation(force_recreate=force_recreate):
        preparator.logger.info("Creating dataset...")
        
        # Create dataset
        dataset_dict, metadata = preparator.create_dataset()
        
        # Save config hash
        preparator.save_config_hash()
        
        print("✅ Data preparation completed successfully!")
        print(f"📁 Dataset saved to: {preparator.data_dir}")
        print(f"📊 Metadata: {metadata}")
        return True
    else:
        print("ℹ️  Using existing files - no recreation needed.")
        print("💡 Use force_recreate=True to recreate all files.")
        return True

def main():
    """Main function to prepare the dataset."""
    parser = argparse.ArgumentParser(description="Prepare English book classification dataset")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force recreation of all files even if they exist")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    success = prepare_data(
        force_recreate=args.force, 
        config_path=args.config
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 