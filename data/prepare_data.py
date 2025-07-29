"""
Data preparation module for The Institutional Books Corpus.
Handles downloading, preprocessing, and splitting data for book sentence classification.
"""

import os
import json
import logging
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

class InstitutionalBooksCorpusProcessor:
    """Processor for The Institutional Books Corpus."""
    
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
        
    def download_corpus(self) -> None:
        """Download The Institutional Books Corpus from Hugging Face."""
        # Load the dataset from Hugging Face (IsmaelMousa/books)
        self.logger.info("Loading IsmaelMousa/books dataset from Hugging Face...")
        try:
            from datasets import load_dataset
            # Load the full dataset (train + validation)
            train_ds = load_dataset("IsmaelMousa/books", split="train")
            val_ds = load_dataset("IsmaelMousa/books", split="validation")
            dataset = list(train_ds) + list(val_ds)

            # The four books to select
            selected_titles = [
                "Anna Karenina",
                "The Adventures of Alice in Wonderland",
                "Frankenstein",
                "The Life of Julius Caesar"
            ]
            corpus_data = {}
            self.logger.info(f"Filtering for selected books: {selected_titles}")
            for row in dataset:
                title = row.get('title', '')
                if title in selected_titles:
                    if title not in corpus_data:
                        # Extract sentences from the EN field
                        text = row.get('EN', '')
                        sentences = sent_tokenize(text)
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
        
    def create_self_supervised_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Create self-supervised learning datasets."""
        self.logger.info("Creating self-supervised learning datasets...")
        
        # MLM dataset (Masked Language Modeling)
        mlm_data = self._create_mlm_dataset(dataset_dict['train'])
        
        # NSP dataset (Next Sentence Prediction)
        nsp_data = self._create_nsp_dataset(dataset_dict['train'])
        
        # Sentence similarity dataset
        similarity_data = self._create_similarity_dataset(dataset_dict['train'])
        
        # Combine all self-supervised datasets
        ss_dataset = DatasetDict({
            'mlm': mlm_data,
            'nsp': nsp_data,
            'similarity': similarity_data
        })
        
        ss_path = self.data_dir / "self_supervised_dataset"
        ss_dataset.save_to_disk(str(ss_path))
        
        self.logger.info(f"Self-supervised datasets saved to {ss_path}")
        
        return ss_dataset
        
    def _create_mlm_dataset(self, train_dataset: Dataset) -> Dataset:
        """Create masked language modeling dataset."""
        # This is a simplified version - in practice, you'd use tokenizer
        mlm_examples = []
        
        for example in train_dataset:
            sentence = example['sentence']
            # Create masked version (simplified)
            words = sentence.split()
            if len(words) > 3:
                mask_idx = np.random.randint(0, len(words))
                masked_words = words.copy()
                masked_words[mask_idx] = '[MASK]'
                
                mlm_examples.append({
                    'original_sentence': sentence,
                    'masked_sentence': ' '.join(masked_words),
                    'masked_word': words[mask_idx],
                    'mask_position': mask_idx
                })
                
        return Dataset.from_list(mlm_examples)
        
    def _create_nsp_dataset(self, train_dataset: Dataset) -> Dataset:
        """Create next sentence prediction dataset."""
        nsp_examples = []
        sentences = train_dataset['sentence']
        
        for i in range(len(sentences) - 1):
            # 50% chance of next sentence being the actual next sentence
            if np.random.random() > 0.5:
                next_sentence = sentences[i + 1]
                is_next = True
            else:
                # Random sentence from different book
                random_idx = np.random.randint(0, len(sentences))
                next_sentence = sentences[random_idx]
                is_next = False
                
            nsp_examples.append({
                'sentence_a': sentences[i],
                'sentence_b': next_sentence,
                'is_next': is_next
            })
            
        return Dataset.from_list(nsp_examples)
        
    def _create_similarity_dataset(self, train_dataset: Dataset) -> Dataset:
        """Create sentence similarity dataset."""
        similarity_examples = []
        sentences = train_dataset['sentence']
        labels = train_dataset['label']
        
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            # Find sentences from same book (positive pairs)
            same_book_indices = [j for j, l in enumerate(labels) if l == label and j != i]
            
            if same_book_indices:
                pos_idx = int(np.random.choice(same_book_indices))
                similarity_examples.append({
                    'sentence_1': sentence,
                    'sentence_2': sentences[pos_idx],
                    'similarity': 1.0
                })
                
            # Find sentences from different books (negative pairs)
            diff_book_indices = [j for j, l in enumerate(labels) if l != label]
            
            if diff_book_indices:
                neg_idx = int(np.random.choice(diff_book_indices))
                similarity_examples.append({
                    'sentence_1': sentence,
                    'sentence_2': sentences[neg_idx],
                    'similarity': 0.0
                })
                
        return Dataset.from_list(similarity_examples)

def main():
    """Main function to prepare the dataset."""
    processor = InstitutionalBooksCorpusProcessor()
    
    # Create main dataset
    dataset_dict, metadata = processor.create_dataset()
    
    # Create self-supervised datasets
    ss_dataset = processor.create_self_supervised_data(dataset_dict)
    
    print("Data preparation completed successfully!")
    print(f"Dataset saved to: {processor.data_dir}")
    print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main() 