#!/usr/bin/env python3
"""
Analyze the semantic pairs to find more meaningful similarities.
"""

import json

def analyze_pairs(filename='semantic_pairs.json'):
    """Analyze pairs and show meaningful results"""
    
    print("Loading pairs...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    print(f"Loaded {len(pairs)} pairs")
    
    # Filter for more meaningful pairs (longer sentences)
    meaningful_pairs = []
    for pair in pairs:
        sent1_len = len(pair['sentence1'].split())
        sent2_len = len(pair['sentence2'].split())
        
        # Only include pairs where both sentences have at least 3 words
        if sent1_len >= 3 and sent2_len >= 3:
            meaningful_pairs.append(pair)
    
    print(f"Found {len(meaningful_pairs)} meaningful pairs (≥3 words each)")
    
    # Show top 20 meaningful pairs
    print("\n" + "="*80)
    print("TOP 20 MOST MEANINGFUL SIMILAR SENTENCES (≥3 words each)")
    print("="*80)
    
    for i, pair in enumerate(meaningful_pairs[:20], 1):
        print(f"\n{i}. Similarity: {pair['similarity']:.4f}")
        print(f"   Book 1: {pair['book1']}")
        print(f"   Book 2: {pair['book2']}")
        print(f"   Sentence 1: {pair['sentence1']}")
        print(f"   Sentence 2: {pair['sentence2']}")
        print("-" * 80)
    
    # Show statistics by book pairs
    print("\n" + "="*60)
    print("PAIRS BY BOOK COMBINATION")
    print("="*60)
    
    book_pair_counts = {}
    for pair in meaningful_pairs:
        book_combo = tuple(sorted([pair['book1'], pair['book2']]))
        book_pair_counts[book_combo] = book_pair_counts.get(book_combo, 0) + 1
    
    for (book1, book2), count in sorted(book_pair_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{book1} ↔ {book2}: {count} pairs")
    
    # Show similarity distribution
    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION")
    print("="*60)
    
    similarity_ranges = {
        '0.95-1.00': 0,
        '0.90-0.95': 0,
        '0.85-0.90': 0,
        '0.80-0.85': 0,
        '0.75-0.80': 0,
        '0.70-0.75': 0
    }
    
    for pair in meaningful_pairs:
        sim = pair['similarity']
        if sim >= 0.95:
            similarity_ranges['0.95-1.00'] += 1
        elif sim >= 0.90:
            similarity_ranges['0.90-0.95'] += 1
        elif sim >= 0.85:
            similarity_ranges['0.85-0.90'] += 1
        elif sim >= 0.80:
            similarity_ranges['0.80-0.85'] += 1
        elif sim >= 0.75:
            similarity_ranges['0.75-0.80'] += 1
        else:
            similarity_ranges['0.70-0.75'] += 1
    
    for range_name, count in similarity_ranges.items():
        print(f"{range_name}: {count} pairs")

if __name__ == "__main__":
    analyze_pairs() 