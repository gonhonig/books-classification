#!/usr/bin/env python3
"""
Step 4: Feature Extraction & Dataset Construction (Improved KNN Approach)
Extract features from fine-tuned semantic embedding model using KNN with caching and multi-label belonging.
"""

import torch
import json
import logging
import argparse
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KNNFeatureExtractor:
    """Extract features from fine-tuned semantic embedding model using KNN with caching."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the feature extractor."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.output_dir = Path("data/features_knn")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device first
        self.device = self._get_device()
        
        # Load data
        self._load_data()
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_data(self):
        """Load the processed dataset and metadata."""
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load processed dataset
        from datasets import load_from_disk
        self.dataset = load_from_disk(str(self.data_dir / "processed_dataset"))
        
        # Load fine-tuned model
        self._load_fine_tuned_model()
        
        logger.info(f"Loaded dataset with {len(self.dataset['train'])} training samples")
        logger.info(f"Books: {self.metadata['books']}")
    
    def _load_fine_tuned_model(self):
        """Load the fine-tuned semantic embedding model."""
        model_path = self.experiments_dir / "semantic_embedding" / "semantic_embedding_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found: {model_path}\n"
                "Please run 'python train_semantic_embedding.py' first to train the model."
            )
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with the same architecture
        from models.semantic_embedding_model import SemanticEmbeddingModel
        self.model = SemanticEmbeddingModel(
            model_name=self.config['model']['encoder']['model_name'],
            embedding_dim=self.config['model']['encoder']['hidden_size']
        )
        
        # Load the trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded fine-tuned model: {checkpoint['model_name']}")
    
    def extract_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Extract embeddings for a list of sentences."""
        logger.info(f"Extracting embeddings for {len(sentences)} sentences...")
        
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting embeddings"):
                batch_sentences = sentences[i:i + batch_size]
                
                # Get embeddings from the fine-tuned model
                outputs = self.model(batch_sentences, batch_sentences)
                batch_embeddings = outputs['embeddings1'].cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_or_create_embeddings_cache(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Get embeddings from cache or create and cache them."""
        cache_path = self.data_dir / "embeddings_cache.npz"
        sentences_cache_path = self.data_dir / "sentences_cache.json"
        labels_cache_path = self.data_dir / "labels_cache.json"
        
        # Check if cache exists
        if cache_path.exists() and sentences_cache_path.exists() and labels_cache_path.exists():
            logger.info("Loading embeddings from cache...")
            
            # Load cached data
            cached_data = np.load(cache_path)
            embeddings = cached_data['embeddings']
            
            with open(sentences_cache_path, 'r') as f:
                sentences = json.load(f)
            
            with open(labels_cache_path, 'r') as f:
                book_labels = json.load(f)
            
            logger.info(f"Loaded {len(embeddings)} cached embeddings")
            return embeddings, sentences, book_labels
        
        # Create embeddings and cache them
        logger.info("Creating embeddings and caching them...")
        
        sentences = []
        book_labels = []
        
        for item in self.dataset['train']:
            sentences.append(item['sentence'])
            book_labels.append(item['book_id'])
        
        # Extract embeddings
        embeddings = self.extract_sentence_embeddings(sentences)
        
        # Save to cache
        np.savez_compressed(cache_path, embeddings=embeddings)
        
        with open(sentences_cache_path, 'w') as f:
            json.dump(sentences, f)
        
        with open(labels_cache_path, 'w') as f:
            json.dump(book_labels, f)
        
        logger.info(f"Cached {len(embeddings)} embeddings for future use")
        
        return embeddings, sentences, book_labels
    
    def compute_knn_similarities(self, test_embeddings: np.ndarray, 
                                train_embeddings: np.ndarray, 
                                train_book_labels: List[str],
                                k_neighbors: int = 5) -> Dict[str, np.ndarray]:
        """Compute KNN-based similarities for each book."""
        logger.info(f"Computing KNN similarities with k={k_neighbors}...")
        
        # Fit KNN on training embeddings
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        knn.fit(train_embeddings)
        
        # Find nearest neighbors for test embeddings
        distances, indices = knn.kneighbors(test_embeddings)
        
        books = self.metadata['books']
        book_similarities = {book: np.zeros(len(test_embeddings)) for book in books}
        
        # For each test sentence, analyze its neighbors
        for i in tqdm(range(len(test_embeddings)), desc="Computing KNN similarities"):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Get book labels of neighbors
            neighbor_books = [train_book_labels[idx] for idx in neighbor_indices]
            neighbor_weights = 1.0 / (1.0 + neighbor_distances)  # Convert distance to similarity
            
            # Count weighted votes for each book
            book_votes = Counter()
            for book, weight in zip(neighbor_books, neighbor_weights):
                book_votes[book] += weight
            
            # Normalize by total votes
            total_votes = sum(book_votes.values())
            if total_votes > 0:
                for book in books:
                    book_similarities[book][i] = book_votes[book] / total_votes
        
        return book_similarities
    
    def compute_multi_label_belonging(self, book_similarities: Dict[str, np.ndarray], 
                                     true_labels: List[str]) -> Dict[str, np.ndarray]:
        """Compute multi-label belonging using the improved logic."""
        logger.info("Computing multi-label belonging...")
        
        books = self.metadata['books']
        book_belonging = {book: np.zeros(len(true_labels)) for book in books}
        
        for i in tqdm(range(len(true_labels)), desc="Computing multi-label belonging"):
            # Get similarity scores for this sentence
            scores = [book_similarities[book][i] for book in books]
            best_score = max(scores)
            true_book = true_labels[i]
            
            # Rule 1: Best score always gets 1
            best_book_idx = scores.index(best_score)
            best_book = books[best_book_idx]
            book_belonging[best_book][i] = 1
            
            # Rule 2: All scores within 0.2 of best get 1
            threshold = best_score - 0.2
            for j, book in enumerate(books):
                if scores[j] >= threshold:
                    book_belonging[book][i] = 1
            
            # Rule 3: Original true book always gets 1
            book_belonging[true_book][i] = 1
        
        return book_belonging
    
    def create_augmented_dataset(self, k_neighbors: int = 5) -> pd.DataFrame:
        """Create augmented dataset using improved KNN approach with multi-label belonging."""
        logger.info("Creating augmented dataset using improved KNN approach...")
        
        # Get all sentences and their labels
        sentences = []
        labels = []
        
        for item in self.dataset['train']:
            sentences.append(item['sentence'])
            labels.append(item['book_id'])
        
        # Get or create embeddings cache
        train_embeddings, train_sentences, train_book_labels = self.get_or_create_embeddings_cache()
        
        # Extract embeddings for all sentences (use cached if same)
        if len(sentences) == len(train_sentences) and sentences == train_sentences:
            all_embeddings = train_embeddings
            logger.info("Using cached embeddings for all sentences")
        else:
            all_embeddings = self.extract_sentence_embeddings(sentences)
        
        # Compute KNN similarities
        book_similarities = self.compute_knn_similarities(
            all_embeddings, train_embeddings, train_book_labels, k_neighbors
        )
        
        # Compute multi-label belonging
        book_belonging = self.compute_multi_label_belonging(book_similarities, labels)
        
        # Create DataFrame
        books = self.metadata['books']
        df_data = []
        
        for i, sentence in enumerate(sentences):
            row = {
                'sentence': sentence,
                'true_book': labels[i]
            }
            
            # Add similarity features for each book
            for book in books:
                row[f'similarity_{book}'] = book_similarities[book][i]
            
            # Add multi-label belonging features
            for book in books:
                row[f'belongs_to_{book}'] = int(book_belonging[book][i])
            
            # Add KNN metadata
            book_scores = [(book, book_similarities[book][i]) for book in books]
            best_book = max(book_scores, key=lambda x: x[1])
            row['knn_best_book'] = best_book[0]
            row['knn_best_score'] = best_book[1]
            
            # Count how many books this sentence belongs to
            row['num_books_belongs_to'] = sum(book_belonging[book][i] for book in books)
            
            # Add confidence score (difference between best and second best)
            sorted_scores = sorted(book_scores, key=lambda x: x[1], reverse=True)
            if len(sorted_scores) >= 2:
                row['knn_confidence'] = sorted_scores[0][1] - sorted_scores[1][1]
            else:
                row['knn_confidence'] = 0.0
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save augmented dataset
        output_path = self.output_dir / "augmented_dataset.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created augmented dataset with {len(df)} samples")
        logger.info(f"Features: {len(books)} similarity scores + {len(books)} multi-label belonging + KNN metadata")
        logger.info(f"Saved to: {output_path}")
        
        return df
    
    def create_feature_summary_improved(self, df: pd.DataFrame) -> Dict:
        """Create a summary of the extracted features using improved approach."""
        books = self.metadata['books']
        
        summary = {
            'total_sentences': len(df),
            'books': books,
            'feature_columns': [col for col in df.columns if col.startswith('similarity_') or col.startswith('belongs_to_') or col.startswith('knn_')],
            'similarity_stats': {},
            'classification_stats': {},
            'multi_label_stats': {},
            'knn_stats': {}
        }
        
        # Similarity statistics
        for book in books:
            similarity_col = f'similarity_{book}'
            summary['similarity_stats'][book] = {
                'mean': float(df[similarity_col].mean()),
                'std': float(df[similarity_col].std()),
                'min': float(df[similarity_col].min()),
                'max': float(df[similarity_col].max())
            }
        
        # Multi-label classification statistics
        for book in books:
            belongs_col = f'belongs_to_{book}'
            summary['classification_stats'][book] = {
                'positive_samples': int(df[belongs_col].sum()),
                'negative_samples': int(len(df) - df[belongs_col].sum()),
                'positive_ratio': float(df[belongs_col].mean())
            }
        
        # Multi-label statistics
        summary['multi_label_stats'] = {
            'mean_books_per_sentence': float(df['num_books_belongs_to'].mean()),
            'single_book_ratio': float((df['num_books_belongs_to'] == 1).mean()),
            'multi_book_ratio': float((df['num_books_belongs_to'] > 1).mean()),
            'max_books_per_sentence': int(df['num_books_belongs_to'].max()),
            'min_books_per_sentence': int(df['num_books_belongs_to'].min())
        }
        
        # KNN statistics
        summary['knn_stats'] = {
            'accuracy': float((df['knn_best_book'] == df['true_book']).mean()),
            'mean_confidence': float(df['knn_confidence'].mean()),
            'high_confidence_ratio': float((df['knn_confidence'] > 0.2).mean()),
            'book_distribution': df['knn_best_book'].value_counts().to_dict()
        }
        
        # Save summary
        summary_path = self.output_dir / "feature_summary_improved.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Feature summary saved to: {summary_path}")
        
        return summary

def extract_features_knn(config_path: str = "configs/config.yaml", k_neighbors: int = 5):
    """Extract features from fine-tuned model using KNN approach."""
    extractor = KNNFeatureExtractor(config_path=config_path)
    
    # Create augmented dataset
    df = extractor.create_augmented_dataset(k_neighbors=k_neighbors)
    
    # Create feature summary
    summary = extractor.create_feature_summary_improved(df)
    
    print(f"\n✅ KNN Feature extraction completed!")
    print(f"📁 Results saved to: {extractor.output_dir}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Features extracted: {len(summary['feature_columns'])}")
    print(f"🎯 KNN Accuracy: {summary['knn_stats']['accuracy']:.4f}")
    print(f"📊 Mean Confidence: {summary['knn_stats']['mean_confidence']:.4f}")
    print(f"📚 Mean Books per Sentence: {summary['multi_label_stats']['mean_books_per_sentence']:.2f}")
    print(f"🔗 Multi-book Ratio: {summary['multi_label_stats']['multi_book_ratio']:.4f}")
    
    return df, summary

def main():
    """Main function to extract features using KNN approach."""
    parser = argparse.ArgumentParser(description="Extract features using KNN approach")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--k-neighbors", "-k", type=int, default=5,
                       help="Number of nearest neighbors for KNN")
    
    args = parser.parse_args()
    
    try:
        df, summary = extract_features_knn(config_path=args.config, k_neighbors=args.k_neighbors)
        print("KNN Feature extraction completed successfully!")
    except Exception as e:
        logger.error(f"KNN Feature extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 