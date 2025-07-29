import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import logging
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic analyzer with pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.sentences = None
        self.book_labels = None
        
    def load_data(self, data_path: str = 'data/processed_dataset'):
        """
        Load processed dataset and extract sentences with book labels.
        
        Args:
            data_path: Path to the processed dataset
        """
        from datasets import load_from_disk
        
        logger.info(f"Loading dataset from {data_path}")
        dataset = load_from_disk(data_path)
        
        # Extract sentences and labels
        self.sentences = []
        self.book_labels = []
        
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                split_data = dataset[split]
                self.sentences.extend(split_data['sentence'])
                self.book_labels.extend(split_data['label'])
        
        logger.info(f"Loaded {len(self.sentences)} sentences")
        
    def compute_embeddings(self):
        """Compute embeddings for all sentences."""
        logger.info("Computing sentence embeddings...")
        self.embeddings = self.model.encode(
            self.sentences, 
            show_progress_bar=True,
            batch_size=32
        )
        logger.info(f"Computed embeddings with shape: {self.embeddings.shape}")
        
    def find_cross_book_similarities(self, similarity_threshold: float = 0.7) -> List[Tuple]:
        """
        Find semantically similar sentences across different books.
        
        Args:
            similarity_threshold: Minimum similarity score to consider sentences similar
            
        Returns:
            List of tuples: (sentence1, sentence2, book1, book2, similarity_score)
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first")
            
        logger.info(f"Finding cross-book similarities with threshold {similarity_threshold}")
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Find cross-book similar pairs
        similar_pairs = []
        book_names = ['The Life of Julius Caesar', 'The Adventures of Alice in Wonderland', 
                     'Anna Karenina', 'Frankenstein']
        
        for i in tqdm(range(len(self.sentences)), desc="Finding similar pairs"):
            for j in range(i + 1, len(self.sentences)):
                # Only consider cross-book pairs
                if self.book_labels[i] != self.book_labels[j]:
                    similarity = similarity_matrix[i][j]
                    if similarity >= similarity_threshold:
                        similar_pairs.append({
                            'sentence1': self.sentences[i],
                            'sentence2': self.sentences[j],
                            'book1': book_names[self.book_labels[i]],
                            'book2': book_names[self.book_labels[j]],
                            'similarity': similarity
                        })
        
        # Sort by similarity score
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(similar_pairs)} cross-book similar pairs")
        return similar_pairs
    
    def analyze_book_specificity(self) -> Dict[str, Any]:
        """
        Analyze how book-specific vs generic each sentence is.
        
        Returns:
            Dictionary with analysis results
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first")
            
        logger.info("Analyzing book specificity...")
        
        # Calculate average similarity within each book
        book_names = ['The Life of Julius Caesar', 'The Adventures of Alice in Wonderland', 
                     'Anna Karenina', 'Frankenstein']
        
        specificity_scores = []
        similarity_matrix = cosine_similarity(self.embeddings)
        
        for i, sentence in enumerate(self.sentences):
            book_label = self.book_labels[i]
            
            # Find sentences from same book
            same_book_indices = [j for j, label in enumerate(self.book_labels) 
                               if label == book_label and j != i]
            
            # Find sentences from different books
            diff_book_indices = [j for j, label in enumerate(self.book_labels) 
                               if label != book_label]
            
            if same_book_indices and diff_book_indices:
                # Average similarity within same book
                same_book_similarity = np.mean([similarity_matrix[i][j] for j in same_book_indices])
                
                # Average similarity to other books
                diff_book_similarity = np.mean([similarity_matrix[i][j] for j in diff_book_indices])
                
                # Specificity score: higher means more book-specific
                specificity = same_book_similarity - diff_book_similarity
                
                specificity_scores.append({
                    'sentence': sentence,
                    'book': book_names[book_label],
                    'same_book_similarity': same_book_similarity,
                    'diff_book_similarity': diff_book_similarity,
                    'specificity': specificity
                })
        
        # Sort by specificity
        specificity_scores.sort(key=lambda x: x['specificity'], reverse=True)
        
        return {
            'specificity_scores': specificity_scores,
            'most_specific': specificity_scores[:10],
            'most_generic': specificity_scores[-10:]
        }
    
    def generate_similarity_report(self, output_path: str = 'data/semantic_analysis.json'):
        """
        Generate comprehensive semantic analysis report.
        
        Args:
            output_path: Path to save the analysis report
        """
        logger.info("Generating semantic analysis report...")
        
        # Find cross-book similarities
        similar_pairs = self.find_cross_book_similarities(similarity_threshold=0.7)
        
        # Analyze book specificity
        specificity_analysis = self.analyze_book_specificity()
        
        # Create report
        report = {
            'model_used': self.model_name,
            'total_sentences': len(self.sentences),
            'cross_book_similar_pairs': similar_pairs[:50],  # Top 50
            'specificity_analysis': specificity_analysis,
            'similarity_threshold': 0.7,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None
        }
        
        # Save report
        with open(output_path, 'w') as f:
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
            
            report_serializable = convert_numpy_types(report)
            json.dump(report_serializable, f, indent=2)
            
        logger.info(f"Report saved to {output_path}")
        
        # Print summary
        print(f"\n=== Semantic Analysis Summary ===")
        print(f"Total sentences: {len(self.sentences)}")
        print(f"Cross-book similar pairs: {len(similar_pairs)}")
        print(f"Most specific sentence: {specificity_analysis['most_specific'][0]['sentence'][:100]}...")
        print(f"Most generic sentence: {specificity_analysis['most_generic'][0]['sentence'][:100]}...")
        
        return report

def main():
    """Run semantic analysis on the prepared dataset."""
    analyzer = SemanticAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Compute embeddings
    analyzer.compute_embeddings()
    
    # Generate analysis report
    report = analyzer.generate_similarity_report()
    
    print("\n=== Top Cross-Book Similar Pairs ===")
    for i, pair in enumerate(report['cross_book_similar_pairs'][:5]):
        print(f"{i+1}. Similarity: {pair['similarity']:.3f}")
        print(f"   Book 1 ({pair['book1']}): {pair['sentence1'][:80]}...")
        print(f"   Book 2 ({pair['book2']}): {pair['sentence2'][:80]}...")
        print()

if __name__ == "__main__":
    main() 