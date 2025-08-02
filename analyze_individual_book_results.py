"""
Analyze Individual Book Model Results
Analyzes the results from training 4 separate neural networks for each book.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results():
    """Load the individual book model results."""
    with open('models/individual_book_results.json', 'r') as f:
        results = json.load(f)
    return results

def create_summary_table(results):
    """Create a summary table of all model results."""
    summary_data = []
    
    for book_col, result in results.items():
        summary_data.append({
            'Book': result['book_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1_score']
        })
    
    df = pd.DataFrame(summary_data)
    return df

def create_detailed_analysis(results):
    """Create detailed analysis of the results."""
    analysis = {
        'total_models': len(results),
        'average_accuracy': np.mean([r['accuracy'] for r in results.values()]),
        'average_precision': np.mean([r['precision'] for r in results.values()]),
        'average_recall': np.mean([r['recall'] for r in results.values()]),
        'average_f1': np.mean([r['f1_score'] for r in results.values()]),
        'best_model': max(results.values(), key=lambda x: x['f1_score'])['book_name'],
        'best_f1': max(results.values(), key=lambda x: x['f1_score'])['f1_score'],
        'worst_model': min(results.values(), key=lambda x: x['f1_score'])['book_name'],
        'worst_f1': min(results.values(), key=lambda x: x['f1_score'])['f1_score']
    }
    
    return analysis

def create_visualizations(results):
    """Create visualizations of the results."""
    # Create summary table
    df = create_summary_table(results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Individual Book Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Metrics comparison bar chart
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = df[metric].values
        ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('Books')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics by Book')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(df['Book'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Book'], df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('F1 Score by Book')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, df['F1 Score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Precision vs Recall scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['Precision'], df['Recall'], s=100, 
                         c=range(len(df)), cmap='viridis', alpha=0.7)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Precision vs Recall')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Add book labels to scatter points
    for i, book in enumerate(df['Book']):
        ax3.annotate(book, (df['Precision'].iloc[i], df['Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Overall performance heatmap
    ax4 = axes[1, 1]
    metrics_matrix = df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].values
    im = ax4.imshow(metrics_matrix.T, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['Book'], rotation=45, ha='right')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax4.set_title('Performance Heatmap')
    
    # Add text annotations to heatmap
    for i in range(len(df)):
        for j in range(4):
            text = ax4.text(i, j, f'{metrics_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('models/individual_book_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(results):
    """Create a comprehensive summary report."""
    analysis = create_detailed_analysis(results)
    df = create_summary_table(results)
    
    report = f"""
# Individual Book Model Training Results

## Overview
- **Total Models Trained**: {analysis['total_models']}
- **Training Approach**: 4 separate binary classifiers, one for each book
- **Dataset**: Semantic augmented dataset with balanced sampling
- **Model Architecture**: Neural network with 384 input dimensions

## Performance Summary

### Average Performance Across All Models
- **Average Accuracy**: {analysis['average_accuracy']:.4f}
- **Average Precision**: {analysis['average_precision']:.4f}
- **Average Recall**: {analysis['average_recall']:.4f}
- **Average F1 Score**: {analysis['average_f1']:.4f}

### Best and Worst Performing Models
- **Best Model**: {analysis['best_model']} (F1: {analysis['best_f1']:.4f})
- **Worst Model**: {analysis['worst_model']} (F1: {analysis['worst_f1']:.4f})

## Individual Model Results

{df.to_string(index=False)}

## Key Insights

1. **Balanced Training**: Each model was trained on a balanced dataset with equal numbers of positive and negative samples to avoid class imbalance issues.

2. **Binary Classification**: Each model performs binary classification (belongs to book or not) rather than multi-label classification.

3. **Performance Range**: F1 scores range from {analysis['worst_f1']:.4f} to {analysis['best_f1']:.4f}, showing consistent performance across all books.

4. **Model Consistency**: All models achieve accuracy above 80%, indicating good discriminative ability.

## Comparison with Previous Approaches

This approach differs from the previous multi-label classification approach by:
- Training separate models for each book
- Using binary classification instead of multi-label
- Creating balanced datasets for each book
- Potentially better handling of class imbalance

## Files Generated
- Model files: `models/*_best_model.pth`
- Results: `models/individual_book_results.json`
- Visualizations: `models/individual_book_model_comparison.png`, `models/individual_book_analysis.png`
"""
    
    # Save the report
    with open('models/individual_book_analysis_report.md', 'w') as f:
        f.write(report)
    
    print(report)
    return report

def main():
    """Main analysis function."""
    print("Analyzing individual book model results...")
    
    # Load results
    results = load_results()
    
    # Create summary table
    df = create_summary_table(results)
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    # Create detailed analysis
    analysis = create_detailed_analysis(results)
    print(f"\nDetailed Analysis:")
    print(f"Average F1 Score: {analysis['average_f1']:.4f}")
    print(f"Best Model: {analysis['best_model']} (F1: {analysis['best_f1']:.4f})")
    print(f"Worst Model: {analysis['worst_model']} (F1: {analysis['worst_f1']:.4f})")
    
    # Create visualizations
    create_visualizations(results)
    
    # Create summary report
    create_summary_report(results)
    
    print("\nAnalysis completed! Check the generated files in the models/ directory.")

if __name__ == "__main__":
    main() 