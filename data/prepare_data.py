"""
Data preparation module for Semantic Book Classification.
Handles downloading, preprocessing, and splitting data for semantic embedding and multi-label classification.
"""

import os
import json
import logging
import argparse
import hashlib
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
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class SemanticBookCorpusProcessor:
    """Processor for Semantic Book Classification Corpus."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the processor with configuration."""
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
        """Generate a hash of the relevant config sections for change detection."""
        # Create a simplified config for hashing (only data-related sections)
        config_for_hash = {
            'data': self.config['data'],
            'model': {
                'encoder': self.config['model']['encoder']
            }
        }
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
        
    def _check_files_exist(self) -> bool:
        """Check if all required files exist."""
        required_files = [
            self.data_dir / "raw_corpus.json",
            self.data_dir / "processed_dataset",
            self.data_dir / "semantic_analysis_data.json",
            self.data_dir / "metadata.json"
        ]
        return all(f.exists() for f in required_files)
        
    def _check_config_changed(self) -> bool:
        """Check if config has changed since last run."""
        config_hash_file = self.data_dir / "config_hash.txt"
        
        current_hash = self._get_config_hash()
        
        if not config_hash_file.exists():
            return True
            
        with open(config_hash_file, 'r') as f:
            stored_hash = f.read().strip()
            
        return current_hash != stored_hash
        
    def _save_config_hash(self):
        """Save current config hash."""
        config_hash_file = self.data_dir / "config_hash.txt"
        current_hash = self._get_config_hash()
        
        with open(config_hash_file, 'w') as f:
            f.write(current_hash)
            
    def needs_recreation(self, force_recreate: bool = False) -> bool:
        """Check if files need to be recreated."""
        if force_recreate:
            self.logger.info("Force recreation requested")
            return True
            
        if not self._check_files_exist():
            self.logger.info("Some required files are missing")
            return True
            
        if self._check_config_changed():
            self.logger.info("Configuration has changed")
            return True
            
        self.logger.info("All files exist and config unchanged - using existing files")
        return False
        
    def download_corpus(self) -> None:
        """Download the book corpus from Hugging Face."""
        # Load the dataset from Hugging Face (IsmaelMousa/books)
        self.logger.info("Loading IsmaelMousa/books dataset from Hugging Face...")
        try:
            from datasets import load_dataset
            # Load the full dataset (train + validation)
            train_ds = load_dataset("IsmaelMousa/books", split="train")
            val_ds = load_dataset("IsmaelMousa/books", split="validation")
            dataset = list(train_ds) + list(val_ds)

            # Get selected books from config
            selected_titles = self.config['data']['selected_books']
            max_sentences = self.config['data'].get('max_sentences_per_book')
            corpus_data = {}
            self.logger.info(f"Filtering for selected books: {selected_titles}")
            if max_sentences:
                self.logger.info(f"Limiting to {max_sentences} sentences per book for testing")
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
                            self.logger.info(f"Limited '{title}' to {max_sentences} sentences (from {len(sent_tokenize(text))})")
                        
                        corpus_data[title] = {
                            "title": title,
                            "author": row.get('author', 'Unknown'),
                            "genre": row.get('category', 'Unknown'),
                            "sentences": sentences
                        }
                        self.logger.info(f"Collected {len(sentences)} sentences for '{title}'")
            # Save the filtered corpus
            corpus_path = self.data_dir / "raw_corpus.json"
            with open(corpus_path, 'w') as f:
                json.dump(corpus_data, f, indent=2)
            self.logger.info(f"Corpus saved to {corpus_path}")
            self.logger.info(f"Total books collected: {len(corpus_data)}")
            for title, data in corpus_data.items():
                self.logger.info(f"  - {title}: {len(data['sentences'])} sentences")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            self.logger.info("Falling back to sample corpus...")
        
    def load_corpus(self) -> Dict:
        """Load the corpus from file."""
        corpus_path = self.data_dir / "raw_corpus.json"
        
        if not corpus_path.exists():
            self.download_corpus()
            
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
            
        return corpus
        
    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """Preprocess sentences for training."""
        processed = []
        
        for sentence in sentences:
            # Clean and normalize
            sentence = sentence.strip()
            
            # Filter by length
            if (len(sentence) >= self.config['data']['min_sentence_length'] and 
                len(sentence) <= self.config['data']['max_sentence_length']):
                processed.append(sentence)
                
        return processed
        
    def create_dataset(self) -> DatasetDict:
        """Create the final dataset for training."""
        self.logger.info("Creating dataset...")
        
        # Load corpus
        corpus = self.load_corpus()
        
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
            'total_size': len(df)
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
        
    def create_semantic_analysis_data(self, dataset_dict: DatasetDict) -> Dict:
        """Create semantic analysis data for multi-label training."""
        self.logger.info("Creating semantic analysis data...")
        
        # Import semantic analysis
        import sys
        sys.path.append('.')
        from utils.semantic_analysis import SemanticAnalyzer
        
        # Run semantic analysis
        analyzer = SemanticAnalyzer()
        analyzer.sentences = []
        analyzer.book_labels = []
        
        # Extract all sentences and labels
        for split in ['train', 'validation', 'test']:
            if split in dataset_dict:
                split_data = dataset_dict[split]
                analyzer.sentences.extend(split_data['sentence'])
                analyzer.book_labels.extend(split_data['label'])
        
        # Compute embeddings
        analyzer.compute_embeddings()
        
        # Find cross-book similarities
        similar_pairs = analyzer.find_cross_book_similarities(
            similarity_threshold=self.config['data']['semantic_analysis']['similarity_threshold']
        )
        
        # Analyze book specificity
        specificity_analysis = analyzer.analyze_book_specificity()
        
        # Create training signals for multi-label classification
        training_signals = self._create_multi_label_signals(
            analyzer.sentences, 
            analyzer.book_labels, 
            similar_pairs,
            specificity_analysis
        )
        
        # Save semantic analysis data
        semantic_data = {
            'similar_pairs': similar_pairs,
            'specificity_analysis': specificity_analysis,
            'training_signals': training_signals,
            'embeddings': analyzer.embeddings.tolist() if analyzer.embeddings is not None else None
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        semantic_data_serializable = convert_numpy_types(semantic_data)
        
        semantic_path = self.data_dir / "semantic_analysis_data.json"
        with open(semantic_path, 'w') as f:
            json.dump(semantic_data_serializable, f, indent=2)
            
        self.logger.info(f"Semantic analysis data saved to {semantic_path}")
        
        return semantic_data
        
    def _create_multi_label_signals(self, sentences: List[str], labels: List[int], 
                                   similar_pairs: List[Dict], specificity_analysis: Dict) -> List[Dict]:
        """Create multi-label training signals based on semantic similarity."""
        book_names = ['The Life of Julius Caesar', 'The Adventures of Alice in Wonderland', 
                     'Anna Karenina', 'Frankenstein']
        
        training_signals = []
        
        for i, (sentence, original_label) in enumerate(zip(sentences, labels)):
            # Initialize book scores
            book_scores = [0.0] * len(book_names)
            
            # Base score for original book
            book_scores[original_label] = 1.0
            
            # Add scores based on semantic similarity to other books
            for pair in similar_pairs:
                if pair['sentence1'] == sentence:
                    # Find which book the similar sentence belongs to
                    similar_sentence = pair['sentence2']
                    similar_book = pair['book2']
                    similarity_score = pair['similarity']
                    
                    # Find book index
                    book_idx = book_names.index(similar_book)
                    book_scores[book_idx] = max(book_scores[book_idx], similarity_score)
                    
                elif pair['sentence2'] == sentence:
                    # Find which book the similar sentence belongs to
                    similar_sentence = pair['sentence1']
                    similar_book = pair['book1']
                    similarity_score = pair['similarity']
                    
                    # Find book index
                    book_idx = book_names.index(similar_book)
                    book_scores[book_idx] = max(book_scores[book_idx], similarity_score)
            
            training_signals.append({
                'sentence': sentence,
                'original_book': book_names[original_label],
                'book_scores': book_scores,
                'is_generic': any(score > 0.5 for score in book_scores if score < 1.0)
            })
        
        return training_signals

def main():
    """Main function to prepare the dataset."""
    parser = argparse.ArgumentParser(description="Prepare semantic book classification dataset")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force recreation of all files even if they exist")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    processor = SemanticBookCorpusProcessor(config_path=args.config)
    
    # Check if recreation is needed
    if processor.needs_recreation(force_recreate=args.force):
        processor.logger.info("Creating dataset...")
        
        # Create main dataset
        dataset_dict, metadata = processor.create_dataset()
        
        # Create semantic analysis data
        semantic_data = processor.create_semantic_analysis_data(dataset_dict)
        
        # Save config hash
        processor._save_config_hash()
        
        print("Data preparation completed successfully!")
        print(f"Dataset saved to: {processor.data_dir}")
        print(f"Metadata: {metadata}")
        print(f"Cross-book similar pairs found: {len(semantic_data['similar_pairs'])}")
        print(f"Multi-label training signals created: {len(semantic_data['training_signals'])}")
    else:
        print("Using existing files - no recreation needed.")
        print("Use --force to recreate all files.")

if __name__ == "__main__":
    main() 