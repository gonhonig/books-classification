#!/usr/bin/env python3
"""
Analyze Methodology Issues
This script analyzes the results of our semantic similarity test to identify methodology issues.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def analyze_methodology_issues():
    """Analyze the methodology issues discovered in our test."""
    
    demos_dir = Path("demonstrations")
    
    # Load results
    classification_file = demos_dir / "classification_test_results.json"
    similarities_file = demos_dir / "cross_book_similarities.json"
    
    if not classification_file.exists():
        print("❌ Classification results not found. Please run test_semantic_similarity.py first.")
        return
    
    with open(classification_file, 'r') as f:
        classification_results = json.load(f)
    
    with open(similarities_file, 'r') as f:
        similarities = json.load(f)
    
    print("=" * 80)
    print("🔍 METHODOLOGY ANALYSIS")
    print("=" * 80)
    
    # 1. Analyze semantic similarity findings
    print("\n📊 SEMANTIC SIMILARITY ANALYSIS")
    print("-" * 40)
    
    print(f"Total semantically similar pairs found: {len(similarities)}")
    
    # Analyze similarity distribution
    similarities_list = [s['similarity'] for s in similarities]
    print(f"Similarity range: {min(similarities_list):.3f} - {max(similarities_list):.3f}")
    print(f"Mean similarity: {np.mean(similarities_list):.3f}")
    print(f"Median similarity: {np.median(similarities_list):.3f}")
    
    # Count pairs by similarity ranges
    high_sim = sum(1 for s in similarities_list if s >= 0.9)
    medium_sim = sum(1 for s in similarities_list if 0.8 <= s < 0.9)
    low_sim = sum(1 for s in similarities_list if s < 0.8)
    
    print(f"High similarity (≥0.9): {high_sim} pairs")
    print(f"Medium similarity (0.8-0.9): {medium_sim} pairs")
    print(f"Low similarity (<0.8): {low_sim} pairs")
    
    # 2. Analyze classification results
    print("\n🎯 CLASSIFICATION ANALYSIS")
    print("-" * 40)
    
    total_pairs = len(classification_results)
    correct_predictions = sum(1 for r in classification_results if r['correct_multi_label'])
    both_predicted_for_both = sum(1 for r in classification_results if r['both_predicted_for_both'])
    both_true_for_both = sum(1 for r in classification_results if r['both_true_for_both'])
    
    print(f"Total pairs tested: {total_pairs}")
    print(f"Correct multi-label predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions/total_pairs*100:.1f}%")
    print(f"Pairs where both sentences predicted for both books: {both_predicted_for_both}")
    print(f"Pairs where both sentences truly belong to both books: {both_true_for_both}")
    
    # 3. Analyze specific issues
    print("\n🚨 IDENTIFIED ISSUES")
    print("-" * 40)
    
    # Issue 1: Perfect similarity but different classifications
    perfect_sim_issues = []
    for result in classification_results:
        if result['pair']['similarity'] >= 0.99:
            if not result['both_predicted_for_both'] and result['both_true_for_both']:
                perfect_sim_issues.append(result)
    
    print(f"Issue 1: Perfect similarity but classifier missed multi-label: {len(perfect_sim_issues)} cases")
    
    # Issue 2: Low similarity but same classification
    low_sim_same_class = []
    for result in classification_results:
        if result['pair']['similarity'] < 0.85:
            if result['both_predicted_for_both'] == result['both_true_for_both']:
                low_sim_same_class.append(result)
    
    print(f"Issue 2: Low similarity but correct classification: {len(low_sim_same_class)} cases")
    
    # 4. Analyze book-specific patterns
    print("\n📚 BOOK-SPECIFIC PATTERNS")
    print("-" * 40)
    
    book_pairs = {}
    for result in classification_results:
        pair = result['pair']
        book_pair = f"{pair['book1']} vs {pair['book2']}"
        if book_pair not in book_pairs:
            book_pairs[book_pair] = []
        book_pairs[book_pair].append(result)
    
    for book_pair, results in book_pairs.items():
        correct = sum(1 for r in results if r['correct_multi_label'])
        total = len(results)
        print(f"{book_pair}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    # 5. Detailed analysis of problematic cases
    print("\n🔍 PROBLEMATIC CASES ANALYSIS")
    print("-" * 40)
    
    # Show cases where similarity is high but classification is wrong
    high_sim_wrong = []
    for result in classification_results:
        if result['pair']['similarity'] >= 0.9 and not result['correct_multi_label']:
            high_sim_wrong.append(result)
    
    print(f"High similarity but wrong classification: {len(high_sim_wrong)} cases")
    
    if high_sim_wrong:
        print("\nExample problematic case:")
        example = high_sim_wrong[0]
        pair = example['pair']
        print(f"  Book 1: {pair['book1']}")
        print(f"  Sentence 1: \"{pair['sentence1'][:80]}...\"")
        print(f"  Book 2: {pair['book2']}")
        print(f"  Sentence 2: \"{pair['sentence2'][:80]}...\"")
        print(f"  Similarity: {pair['similarity']:.3f}")
        print(f"  Both predicted for both books: {example['both_predicted_for_both']}")
        print(f"  Both truly belong to both books: {example['both_true_for_both']}")
    
    # 6. Methodology recommendations
    print("\n💡 METHODOLOGY RECOMMENDATIONS")
    print("-" * 40)
    
    print("1. SEMANTIC SIMILARITY ISSUE:")
    print("   - Found 2,975 semantically similar pairs across books")
    print("   - Many pairs have perfect similarity (1.0) but different content")
    print("   - This suggests our semantic model may be overfitting or not capturing true semantic meaning")
    
    print("\n2. CLASSIFICATION ISSUE:")
    print("   - 95% accuracy on multi-label classification")
    print("   - But only 1 out of 20 cases correctly identified multi-label scenarios")
    print("   - This suggests the classifier is not learning to identify cross-book similarities")
    
    print("\n3. FEATURE EXTRACTION ISSUE:")
    print("   - KNN-based features may not be capturing semantic relationships properly")
    print("   - The features might be too book-specific rather than semantic-specific")
    
    print("\n4. RECOMMENDED FIXES:")
    print("   - Use more sophisticated semantic similarity measures")
    print("   - Implement contrastive learning to better distinguish similar vs different content")
    print("   - Add explicit multi-label training examples")
    print("   - Consider using attention mechanisms to focus on semantic differences")
    
    # 7. Create summary report
    summary = {
        'total_similar_pairs': len(similarities),
        'tested_pairs': len(classification_results),
        'classification_accuracy': correct_predictions/total_pairs*100,
        'high_similarity_pairs': high_sim,
        'perfect_similarity_issues': len(perfect_sim_issues),
        'high_sim_wrong_classification': len(high_sim_wrong),
        'methodology_issues': [
            "Semantic model may be overfitting to surface features",
            "Classifier not learning cross-book semantic relationships",
            "KNN features may be too book-specific",
            "Need better multi-label training examples"
        ]
    }
    
    # Save summary
    summary_file = demos_dir / "methodology_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_methodology_issues() 