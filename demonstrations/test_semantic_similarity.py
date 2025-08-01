#!/usr/bin/env python3
"""
Test Semantic Similarity Across Books
This script systematically tests our methodology by:
1. Extracting embeddings from test data
2. Finding semantically close sentences across books
3. Testing if our classifier correctly identifies multi-label cases
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import yaml
from typing import List, Dict, Tuple, Any
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSimilarityTester:
    """Test semantic similarity and classification across books."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the tester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.demos_dir = Path("demonstrations")
        self.demos_dir.mkdir(exist_ok=True)
        
        # Load the semantic model
        model_name = self.config['semantic_models']['selected_model']
        self.semantic_model = SentenceTransformer(model_name)
        logger.info(f"Loaded semantic model: {model_name}")
        
        # Load the dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the dataset with features and labels."""
        features_file = self.data_dir / "features_knn" / "augmented_dataset.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        self.df = pd.read_csv(features_file)
        logger.info(f"Loaded dataset: {self.df.shape}")
        
        # Get book names
        self.books = [col.replace('belongs_to_', '') for col in self.df.columns if col.startswith('belongs_to_')]
        logger.info(f"Books: {self.books}")
        
        # Get test data (last 15% of data)
        test_size = int(len(self.df) * 0.15)
        self.test_df = self.df.tail(test_size).copy()
        logger.info(f"Test data size: {len(self.test_df)}")
        
    def extract_test_embeddings(self):
        """Extract embeddings for all test sentences."""
        logger.info("Extracting embeddings for test sentences...")
        
        test_sentences = self.test_df['sentence'].tolist()
        embeddings = self.semantic_model.encode(test_sentences)
        
        # Save embeddings
        embeddings_file = self.demos_dir / "test_embeddings.npz"
        np.savez_compressed(embeddings_file, embeddings=embeddings)
        
        logger.info(f"Saved test embeddings to: {embeddings_file}")
        return embeddings
    
    def find_semantically_close_sentences(self, similarity_threshold: float = 0.7):
        """Find semantically close sentences across different books."""
        logger.info(f"Finding semantically close sentences (threshold: {similarity_threshold})...")
        
        # Load embeddings
        embeddings_file = self.demos_dir / "test_embeddings.npz"
        if not embeddings_file.exists():
            embeddings = self.extract_test_embeddings()
        else:
            embeddings = np.load(embeddings_file)['embeddings']
        
        # Group sentences by book with local indices
        sentences_by_book = {}
        for book in self.books:
            book_mask = self.test_df[f'belongs_to_{book}'] == 1
            book_sentences = self.test_df[book_mask]['sentence'].tolist()
            # Use local indices (0 to len(test_df)-1) instead of original dataset indices
            book_local_indices = self.test_df[book_mask].index.get_indexer(self.test_df[book_mask].index)
            sentences_by_book[book] = {
                'sentences': book_sentences,
                'local_indices': book_local_indices,
                'original_indices': self.test_df[book_mask].index.tolist()
            }
        
        # Find cross-book similarities
        cross_book_similarities = []
        
        for book1 in self.books:
            for book2 in self.books:
                if book1 >= book2:  # Avoid duplicates
                    continue
                
                logger.info(f"Comparing {book1} vs {book2}...")
                
                # Get sentences and embeddings for both books
                book1_data = sentences_by_book[book1]
                book2_data = sentences_by_book[book2]
                
                if not book1_data['sentences'] or not book2_data['sentences']:
                    continue
                
                # Get embeddings for these sentences using local indices
                book1_embeddings = embeddings[book1_data['local_indices']]
                book2_embeddings = embeddings[book2_data['local_indices']]
                
                # Calculate similarities
                similarities = cosine_similarity(book1_embeddings, book2_embeddings)
                
                # Find high similarity pairs
                for i, (sent1, local_idx1, orig_idx1) in enumerate(zip(book1_data['sentences'], book1_data['local_indices'], book1_data['original_indices'])):
                    for j, (sent2, local_idx2, orig_idx2) in enumerate(zip(book2_data['sentences'], book2_data['local_indices'], book2_data['original_indices'])):
                        similarity = similarities[i, j]
                        
                        if similarity >= similarity_threshold:
                            cross_book_similarities.append({
                                'book1': book1,
                                'book2': book2,
                                'sentence1': sent1,
                                'sentence2': sent2,
                                'similarity': similarity,
                                'index1': orig_idx1,
                                'index2': orig_idx2
                            })
        
        # Sort by similarity
        cross_book_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(cross_book_similarities)} semantically close sentence pairs")
        
        # Save results
        results_file = self.demos_dir / "cross_book_similarities.json"
        
        # Convert numpy types to Python types for JSON serialization
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
        
        cross_book_similarities_converted = convert_numpy_types(cross_book_similarities)
        
        with open(results_file, 'w') as f:
            json.dump(cross_book_similarities_converted, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved cross-book similarities to: {results_file}")
        return cross_book_similarities
    
    def test_classifier_on_similar_sentences(self, similar_sentences: List[Dict]):
        """Test if our classifier correctly identifies multi-label cases."""
        logger.info("Testing classifier on semantically similar sentences...")
        
        # Load the trained classifier
        classifier_file = Path("experiments/multi_label_classifier_knn_corrected/random_forest_model.pkl")
        if not classifier_file.exists():
            logger.error(f"Classifier not found: {classifier_file}")
            return None
        
        with open(classifier_file, 'rb') as f:
            classifier = pickle.load(f)
        
        # Load feature columns info
        metadata_file = Path("experiments/multi_label_classifier_knn_corrected/model_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        feature_columns = metadata['feature_columns']
        
        # Test each similar sentence pair
        classification_results = []
        
        for pair in similar_sentences[:20]:  # Test first 20 pairs
            # Get the sentences from our dataset
            idx1, idx2 = pair['index1'], pair['index2']
            
            if idx1 in self.df.index and idx2 in self.df.index:
                # Get features for both sentences
                features1 = self.df.loc[idx1, feature_columns].values.reshape(1, -1)
                features2 = self.df.loc[idx2, feature_columns].values.reshape(1, -1)
                
                # Predict
                pred1 = classifier.predict(features1)[0]
                pred2 = classifier.predict(features2)[0]
                
                # Get true labels
                true_labels1 = [self.df.loc[idx1, f'belongs_to_{book}'] for book in self.books]
                true_labels2 = [self.df.loc[idx2, f'belongs_to_{book}'] for book in self.books]
                
                # Check if both sentences are predicted as belonging to both books
                book1_idx = self.books.index(pair['book1'])
                book2_idx = self.books.index(pair['book2'])
                
                both_predicted_for_both = (
                    pred1[book1_idx] == 1 and pred1[book2_idx] == 1 and
                    pred2[book1_idx] == 1 and pred2[book2_idx] == 1
                )
                
                both_true_for_both = (
                    true_labels1[book1_idx] == 1 and true_labels1[book2_idx] == 1 and
                    true_labels2[book1_idx] == 1 and true_labels2[book2_idx] == 1
                )
                
                classification_results.append({
                    'pair': pair,
                    'pred1': pred1.tolist(),
                    'pred2': pred2.tolist(),
                    'true1': true_labels1,
                    'true2': true_labels2,
                    'both_predicted_for_both': both_predicted_for_both,
                    'both_true_for_both': both_true_for_both,
                    'correct_multi_label': both_predicted_for_both == both_true_for_both
                })
        
        # Save classification results
        results_file = self.demos_dir / "classification_test_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        classification_results_converted = convert_numpy_types(classification_results)
        
        with open(results_file, 'w') as f:
            json.dump(classification_results_converted, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved classification test results to: {results_file}")
        
        # Print summary
        self._print_classification_summary(classification_results)
        
        return classification_results
    
    def _print_classification_summary(self, results: List[Dict]):
        """Print a summary of classification results."""
        print("\n" + "="*80)
        print("CLASSIFICATION TEST RESULTS")
        print("="*80)
        
        total_pairs = len(results)
        correct_multi_label = sum(1 for r in results if r['correct_multi_label'])
        both_predicted_for_both = sum(1 for r in results if r['both_predicted_for_both'])
        both_true_for_both = sum(1 for r in results if r['both_true_for_both'])
        
        print(f"Total pairs tested: {total_pairs}")
        print(f"Pairs where both sentences predicted for both books: {both_predicted_for_both}")
        print(f"Pairs where both sentences truly belong to both books: {both_true_for_both}")
        print(f"Correct multi-label predictions: {correct_multi_label}")
        print(f"Accuracy: {correct_multi_label/total_pairs*100:.1f}%")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results[:5]):  # Show first 5
            pair = result['pair']
            print(f"\nPair {i+1}:")
            print(f"  {pair['book1']}: \"{pair['sentence1'][:60]}...\"")
            print(f"  {pair['book2']}: \"{pair['sentence2'][:60]}...\"")
            print(f"  Similarity: {pair['similarity']:.3f}")
            print(f"  Both predicted for both books: {result['both_predicted_for_both']}")
            print(f"  Both truly belong to both books: {result['both_true_for_both']}")
            print(f"  Correct: {result['correct_multi_label']}")
        
        print("\n" + "="*80)
    
    def run_complete_test(self):
        """Run the complete semantic similarity and classification test."""
        logger.info("Starting complete semantic similarity and classification test...")
        
        # Step 1: Extract embeddings
        embeddings = self.extract_test_embeddings()
        
        # Step 2: Find semantically close sentences
        similar_sentences = self.find_semantically_close_sentences(similarity_threshold=0.8)
        
        # Step 3: Test classifier
        classification_results = self.test_classifier_on_similar_sentences(similar_sentences)
        
        logger.info("Complete test finished!")
        return {
            'embeddings': embeddings,
            'similar_sentences': similar_sentences,
            'classification_results': classification_results
        }

def main():
    """Main function to run the semantic similarity test."""
    tester = SemanticSimilarityTester()
    results = tester.run_complete_test()
    
    print("\nTest completed successfully!")
    print(f"Results saved to: {tester.demos_dir}")

if __name__ == "__main__":
    main() 