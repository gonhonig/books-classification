#!/usr/bin/env python3
"""
Create semantic analysis data from similarity test pairs for training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_semantic_analysis_data():
    """Create semantic analysis data from similarity test pairs."""
    
    # Load similarity test pairs
    with open("data/similarity_test_pairs.json", 'r') as f:
        test_pairs = json.load(f)
    
    # Extract similar pairs (type == "similar")
    similar_pairs = []
    all_sentences = set()
    
    for pair in test_pairs:
        if pair['type'] == 'similar':
            similar_pairs.append({
                'sentence1': pair['sentence1'],
                'sentence2': pair['sentence2'],
                'similarity': pair['similarity'],
                'source_book': pair['source_book']
            })
            all_sentences.add(pair['sentence1'])
            all_sentences.add(pair['sentence2'])
    
    # Create training signals (all unique sentences)
    training_signals = []
    for sentence in all_sentences:
        training_signals.append({
            'sentence': sentence,
            'type': 'training_signal'
        })
    
    # Create semantic analysis data
    semantic_data = {
        'similar_pairs': similar_pairs,
        'training_signals': training_signals,
        'total_similar_pairs': len(similar_pairs),
        'total_sentences': len(all_sentences)
    }
    
    # Save to file
    output_path = Path("data/semantic_analysis_data.json")
    with open(output_path, 'w') as f:
        json.dump(semantic_data, f, indent=2)
    
    logger.info(f"Created semantic analysis data:")
    logger.info(f"  Similar pairs: {len(similar_pairs)}")
    logger.info(f"  Unique sentences: {len(all_sentences)}")
    logger.info(f"  Saved to: {output_path}")
    
    return semantic_data

if __name__ == "__main__":
    create_semantic_analysis_data() 