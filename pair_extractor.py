#!/usr/bin/env python3
"""
Simple pair extractor for semantic similarity analysis.
Extracts embeddings, compares sentences from different books, and stores similarity pairs.
"""

import json
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Set
import argparse


def load_raw_corpus():
    """Load the raw corpus from the original data structure"""
    # Try to load deduplicated corpus first
    try:
        with open('data/corpus_deduplicated.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"Loaded deduplicated corpus with {len(corpus)} books")
        return corpus
    except Exception as e:
        print(f"Warning: Could not load corpus_deduplicated.json: {e}")
        
        # Fall back to original corpus
        try:
            with open('data/corpus.json', 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            print(f"Loaded original corpus with {len(corpus)} books")
            return corpus
        except Exception as e:
            print(f"Warning: Could not load corpus.json: {e}")
            print("Warning: Could not load cached data. Using sample data.")
            return {
                "Sample Book 1": {
                    "title": "Sample Book 1",
                    "author": "Sample Author 1", 
                    "genre": "Sample Genre",
                    "sentences": [
                        "The quick brown fox jumps over the lazy dog.",
                        "A lazy dog sleeps while a quick brown fox jumps.",
                        "Alice in Wonderland is a children's book."
                    ]
                },
                "Sample Book 2": {
                    "title": "Sample Book 2", 
                    "author": "Sample Author 2",
                    "genre": "Sample Genre",
                    "sentences": [
                        "The children's book tells the story of Alice.",
                        "A fox and a dog are common characters in stories.",
                        "Wonderland is a magical place."
                    ]
                }
            }


def normalize_sentence(sentence: str) -> str:
    """Normalize sentence for comparison (lowercase and strip)"""
    return sentence.lower().strip()


def extract_sentences_by_book(corpus: Dict) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """
    Extract sentences organized by book with metadata.
    
    Returns:
        - sentences_by_book: Dict[book_title, List[sentences]]
        - book_metadata: Dict[book_title, metadata]
    """
    sentences_by_book = {}
    book_metadata = {}
    
    for book_title, book_data in corpus.items():
        if 'sentences' in book_data:
            # Clean sentences
            cleaned_sentences = []
            for sentence in book_data['sentences']:
                cleaned = ' '.join(sentence.split())
                if cleaned.strip():
                    cleaned_sentences.append(cleaned)
            
            if cleaned_sentences:
                sentences_by_book[book_title] = cleaned_sentences
                book_metadata[book_title] = {
                    'title': book_data.get('title', book_title),
                    'author': book_data.get('author', 'Unknown'),
                    'genre': book_data.get('genre', 'Unknown')
                }
    
    print(f"Extracted sentences from {len(sentences_by_book)} books")
    for book, sentences in sentences_by_book.items():
        print(f"  {book}: {len(sentences)} sentences")
    
    return sentences_by_book, book_metadata


def get_embeddings_cache_path(model_name: str) -> str:
    """Get the cache path for embeddings"""
    model_hash = hashlib.md5(model_name.encode()).hexdigest()
    return f'data/embeddings_cache_{model_hash}.npz'


def extract_embeddings_cached(sentences_by_book: Dict[str, List[str]], 
                            model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Extract embeddings with caching and return embeddings + book index mapping.
    
    Returns:
        - embeddings: np.ndarray of all sentence embeddings
        - book_indices: Dict[book_title, (start_idx, end_idx)]
    """
    cache_path = get_embeddings_cache_path(model_name)
    
    # Flatten all sentences and create book index mapping
    all_sentences = []
    book_indices = {}
    current_idx = 0
    
    for book_title, sentences in sentences_by_book.items():
        start_idx = current_idx
        all_sentences.extend(sentences)
        current_idx = len(all_sentences)
        book_indices[book_title] = (start_idx, current_idx)
    
    try:
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        embeddings = data['embeddings']
        print("Using cached embeddings")
        return embeddings, book_indices
    except:
        print(f"Cache not found, computing embeddings with {model_name}...")
        
        # Load model
        model = SentenceTransformer(model_name)
        
        # Compute embeddings
        embeddings = model.encode(all_sentences, show_progress_bar=True, batch_size=32)
        
        # Save to cache
        print(f"Saving embeddings to {cache_path}")
        np.savez_compressed(cache_path, embeddings=embeddings)
        
        return embeddings, book_indices


def compute_similarity_pairs(embeddings: np.ndarray, 
                           sentences_by_book: Dict[str, List[str]], 
                           book_indices: Dict[str, Tuple[int, int]],
                           min_similarity: float = 0.7) -> List[Dict]:
    """
    Compute similarity pairs between sentences from different books.
    
    Args:
        embeddings: Sentence embeddings
        sentences_by_book: Sentences organized by book
        book_indices: Book index mapping
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of similarity pairs with metadata
    """
    print("Computing similarity pairs between different books...")
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    pairs = []
    used_pairs = set()  # Track processed pairs to avoid duplicates
    used_sentence_pairs = set()  # Track sentence content pairs to avoid semantic duplicates
    processed_pairs = 0
    skipped_exact = 0
    skipped_duplicate = 0
    skipped_semantic_duplicate = 0
    
    book_titles = list(sentences_by_book.keys())
    
    # Compare each book with other books
    for i, book1 in enumerate(book_titles):
        start1, end1 = book_indices[book1]
        sentences1 = sentences_by_book[book1]
        
        for j, book2 in enumerate(book_titles):
            if i >= j:  # Skip same book and already processed pairs
                continue
                
            start2, end2 = book_indices[book2]
            sentences2 = sentences_by_book[book2]
            
            print(f"Comparing {book1} ({len(sentences1)} sentences) with {book2} ({len(sentences2)} sentences)")
            
            # Get embeddings for both books
            embeddings1 = embeddings_norm[start1:end1]
            embeddings2 = embeddings_norm[start2:end2]
            
            # Compute similarity matrix
            similarity_matrix = np.dot(embeddings1, embeddings2.T)
            
            # Find pairs above threshold
            for idx1 in range(len(sentences1)):
                for idx2 in range(len(sentences2)):
                    similarity = similarity_matrix[idx1, idx2]
                    
                    if similarity >= min_similarity:
                        # Get global indices
                        global_idx1 = start1 + idx1
                        global_idx2 = start2 + idx2
                        
                        # Create pair identifier (sorted to avoid duplicates)
                        pair_id = tuple(sorted([global_idx1, global_idx2]))
                        
                        # Skip if already processed
                        if pair_id in used_pairs:
                            skipped_duplicate += 1
                            continue
                        
                        # Check for exact same sentence (case-insensitive)
                        sentence1 = sentences1[idx1]
                        sentence2 = sentences2[idx2]
                        if normalize_sentence(sentence1) == normalize_sentence(sentence2):
                            skipped_exact += 1
                            continue
                        
                        # Check for semantic duplicates (same sentence content)
                        sentence_pair = tuple(sorted([normalize_sentence(sentence1), normalize_sentence(sentence2)]))
                        if sentence_pair in used_sentence_pairs:
                            skipped_semantic_duplicate += 1
                            continue
                        
                        # Add pair
                        used_pairs.add(pair_id)
                        used_sentence_pairs.add(sentence_pair)
                        pairs.append({
                            'index1': global_idx1,
                            'index2': global_idx2,
                            'similarity': float(similarity),
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'book1': book1,
                            'book2': book2
                        })
                        processed_pairs += 1
    
    print(f"Found {len(pairs)} similarity pairs")
    print(f"Skipped {skipped_exact} exact matches")
    print(f"Skipped {skipped_duplicate} duplicate pairs")
    print(f"Skipped {skipped_semantic_duplicate} semantic duplicates")
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return pairs


def save_results(pairs: List[Dict], 
                sentences_by_book: Dict[str, List[str]], 
                book_metadata: Dict[str, Dict],
                output_file: str = 'semantic_pairs.json'):
    """Save results to JSON file"""
    print(f"Saving {len(pairs)} pairs to {output_file}")
    
    # Prepare results with metadata
    results = {
        'pairs': pairs,
        'metadata': {
            'total_pairs': len(pairs),
            'total_books': len(sentences_by_book),
            'book_info': book_metadata
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def print_top_pairs(pairs: List[Dict], top_n: int = 10):
    """Print top similarity pairs"""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MOST SIMILAR SENTENCES FROM DIFFERENT BOOKS")
    print(f"{'='*80}")
    
    for i, pair in enumerate(pairs[:top_n], 1):
        print(f"\n{i}. Similarity: {pair['similarity']:.4f}")
        print(f"   Book 1: {pair['book1']}")
        print(f"   Book 2: {pair['book2']}")
        print(f"   Sentence 1: {pair['sentence1'][:100]}{'...' if len(pair['sentence1']) > 100 else ''}")
        print(f"   Sentence 2: {pair['sentence2'][:100]}{'...' if len(pair['sentence2']) > 100 else ''}")
        print("-" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple pair extractor for semantic similarity')
    parser.add_argument('--min_similarity', type=float, default=0.7, 
                       help='Minimum similarity threshold (default: 0.7)')
    parser.add_argument('--output', type=str, default='semantic_pairs.json',
                       help='Output file name (default: semantic_pairs.json)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top pairs to display (default: 10)')
    
    args = parser.parse_args()
    
    print("Simple Pair Extractor for Semantic Similarity")
    print("="*50)
    print(f"Parameters: min_similarity={args.min_similarity}, output={args.output}")
    print("="*50)
    
    # Load corpus
    print("Loading corpus...")
    corpus = load_raw_corpus()
    
    # Extract sentences by book
    print("Extracting sentences by book...")
    sentences_by_book, book_metadata = extract_sentences_by_book(corpus)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, book_indices = extract_embeddings_cached(sentences_by_book)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity pairs
    pairs = compute_similarity_pairs(
        embeddings, sentences_by_book, book_indices, 
        min_similarity=args.min_similarity
    )
    
    # Print top pairs
    print_top_pairs(pairs, top_n=args.top_n)
    
    # Save results
    save_results(pairs, sentences_by_book, book_metadata, args.output)
    
    print(f"\nExtraction completed! Found {len(pairs)} similarity pairs.")


if __name__ == "__main__":
    main() 