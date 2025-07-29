#!/usr/bin/env python3
"""
Improved semantic analysis to find more meaningful similar pairs.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
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

class ImprovedSemanticAnalyzer:
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def load_data(self, data_path="data/processed_dataset"):
        """Load processed dataset."""
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        return dataset
        
    def compute_embeddings(self, sentences, batch_size=32):
        """Compute embeddings for sentences."""
        logger.info(f"Computing embeddings for {len(sentences)} sentences...")
        embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Computing embeddings"):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
    
    def find_meaningful_similarities(self, sentences, book_labels, min_similarity=0.75, max_similarity=0.95):
        """Find meaningful similar pairs with better filtering."""
        logger.info("Finding meaningful similar pairs...")
        
        # Compute embeddings
        embeddings = self.compute_embeddings(sentences)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find similar pairs with better criteria
        similar_pairs = []
        
        for i in tqdm(range(len(sentences)), desc="Finding similar pairs"):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j]
                
                # Skip if not in similarity range
                if similarity < min_similarity or similarity > max_similarity:
                    continue
                
                # Skip if same book
                if book_labels[i] == book_labels[j]:
                    continue
                
                # Skip if sentences are too similar (nearly identical)
                sent1_clean = sentences[i].lower().strip()
                sent2_clean = sentences[j].lower().strip()
                
                # Skip if sentences are nearly identical
                if sent1_clean == sent2_clean:
                    continue
                
                # Skip if one is substring of the other
                if sent1_clean in sent2_clean or sent2_clean in sent1_clean:
                    continue
                
                # Skip very short sentences
                if len(sent1_clean.split()) < 3 or len(sent2_clean.split()) < 3:
                    continue
                
                # Skip sentences that are too long
                if len(sent1_clean.split()) > 20 or len(sent2_clean.split()) > 20:
                    continue
                
                # Additional semantic checks
                if self._is_meaningful_pair(sent1_clean, sent2_clean):
                    similar_pairs.append({
                        'sentence1': sentences[i],
                        'sentence2': sentences[j],
                        'book1': book_labels[i],
                        'book2': book_labels[j],
                        'similarity': float(similarity)
                    })
        
        logger.info(f"Found {len(similar_pairs)} meaningful similar pairs")
        return similar_pairs
    
    def _is_meaningful_pair(self, sent1, sent2):
        """Check if a pair is semantically meaningful."""
        # Skip if they share too many words (likely paraphrases)
        words1 = set(sent1.split())
        words2 = set(sent2.split())
        
        # Calculate word overlap
        overlap = len(words1.intersection(words2))
        total_words = len(words1.union(words2))
        
        # Skip if more than 70% word overlap
        if total_words > 0 and overlap / total_words > 0.7:
            return False
        
        # Skip common phrases that appear in many books
        common_phrases = [
            'she thought', 'he thought', 'she said', 'he said',
            'i am', 'you are', 'it is', 'that is',
            'very well', 'all right', 'of course',
            'poor little', 'dear little', 'my dear'
        ]
        
        for phrase in common_phrases:
            if phrase in sent1 and phrase in sent2:
                return False
        
        return True
    
    def analyze_book_specificity(self, sentences, book_labels, embeddings):
        """Analyze how book-specific each sentence is."""
        logger.info("Analyzing book specificity...")
        
        specificity_scores = []
        
        for i, sentence_embedding in enumerate(embeddings):
            # Calculate average similarity to sentences from same book vs different books
            same_book_similarities = []
            different_book_similarities = []
            
            current_book = book_labels[i]
            
            for j, other_embedding in enumerate(embeddings):
                if i != j:
                    similarity = np.dot(sentence_embedding, other_embedding) / (
                        np.linalg.norm(sentence_embedding) * np.linalg.norm(other_embedding)
                    )
                    
                    if book_labels[j] == current_book:
                        same_book_similarities.append(similarity)
                    else:
                        different_book_similarities.append(similarity)
            
            # Calculate specificity
            if same_book_similarities and different_book_similarities:
                specificity = np.mean(same_book_similarities) - np.mean(different_book_similarities)
            else:
                specificity = 0
            
            specificity_scores.append(specificity)
        
        return specificity_scores
    
    def generate_improved_report(self, sentences, book_labels, similar_pairs, specificity_scores):
        """Generate an improved analysis report."""
        logger.info("Generating improved analysis report...")
        
        # Calculate statistics
        total_sentences = len(sentences)
        unique_books = len(set(book_labels))
        
        # Book distribution
        book_counts = {}
        for book in book_labels:
            book_counts[book] = book_counts.get(book, 0) + 1
        
        # Similarity statistics
        similarities = [pair['similarity'] for pair in similar_pairs]
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Specificity statistics
        avg_specificity = np.mean(specificity_scores)
        
        report = {
            'dataset_stats': {
                'total_sentences': total_sentences,
                'unique_books': unique_books,
                'book_distribution': book_counts
            },
            'similarity_analysis': {
                'total_similar_pairs': len(similar_pairs),
                'average_similarity': avg_similarity,
                'similarity_range': {
                    'min': float(np.min(similarities)) if similarities else 0,
                    'max': float(np.max(similarities)) if similarities else 0
                }
            },
            'specificity_analysis': {
                'average_specificity': float(avg_specificity),
                'specificity_range': {
                    'min': float(np.min(specificity_scores)),
                    'max': float(np.max(specificity_scores))
                }
            },
            'sample_similar_pairs': similar_pairs[:10]  # First 10 pairs as examples
        }
        
        return report

def main():
    """Main function to run improved semantic analysis."""
    # Load data
    analyzer = ImprovedSemanticAnalyzer(similarity_threshold=0.75)
    dataset = analyzer.load_data()
    
    # Extract sentences and labels
    sentences = dataset['train']['sentence']
    book_labels = dataset['train']['book_id']
    
    # Use all sentences for better analysis
    logger.info(f"Using all {len(sentences)} sentences for analysis")
    
    # Find meaningful similar pairs
    similar_pairs = analyzer.find_meaningful_similarities(
        sentences, book_labels, 
        min_similarity=0.65, 
        max_similarity=0.90
    )
    
    # Compute embeddings for specificity analysis
    embeddings = analyzer.compute_embeddings(sentences)
    
    # Analyze book specificity
    specificity_scores = analyzer.analyze_book_specificity(sentences, book_labels, embeddings)
    
    # Generate report
    report = analyzer.generate_improved_report(sentences, book_labels, similar_pairs, specificity_scores)
    
    # Save results
    output_path = "data/improved_semantic_analysis_data.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(report), f, indent=2)
    
    print(f"\n=== IMPROVED SEMANTIC ANALYSIS COMPLETED ===")
    print(f"Total sentences analyzed: {len(sentences)}")
    print(f"Meaningful similar pairs found: {len(similar_pairs)}")
    print(f"Average similarity: {report['similarity_analysis']['average_similarity']:.3f}")
    print(f"Average specificity: {report['specificity_analysis']['average_specificity']:.3f}")
    print(f"\nSample similar pairs:")
    for i, pair in enumerate(report['sample_similar_pairs'][:5]):
        print(f"  {i+1}. \"{pair['sentence1'][:50]}...\" <-> \"{pair['sentence2'][:50]}...\" (sim: {pair['similarity']:.3f})")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main() 