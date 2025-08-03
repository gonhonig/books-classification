"""
Comprehensive Comparison: Multi-Label vs Per-Book Approaches
Analyzes and compares the performance of both approaches across different metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ApproachComparator:
    """Compare multi-label vs per-book approaches."""
    
    def __init__(self):
        self.per_book_results = None
        self.multi_label_results = None
        self.comparison_data = {}
        
    def load_results(self):
        """Load results from both approaches."""
        logger.info("Loading results from both approaches...")
        
        # Load per-book results
        with open('model_per_book/per_book_detailed_metrics.json', 'r') as f:
            self.per_book_results = json.load(f)
        
        # Load multi-label results
        with open('multi_label_model/per_book_detailed_metrics.json', 'r') as f:
            self.multi_label_results = json.load(f)
        
        logger.info("Results loaded successfully")
        
    def calculate_comparison_metrics(self):
        """Calculate comprehensive comparison metrics."""
        logger.info("Calculating comparison metrics...")
        
        books = list(self.per_book_results.keys())
        
        for book in books:
            per_book = self.per_book_results[book]
            multi_label = self.multi_label_results[book]
            
            # Overall performance comparison
            overall_comparison = {
                'per_book': {
                    'accuracy': per_book['overall']['accuracy'],
                    'precision': per_book['overall']['precision'],
                    'recall': per_book['overall']['recall'],
                    'f1': per_book['overall']['f1']
                },
                'multi_label': {
                    'accuracy': multi_label['overall']['accuracy'],
                    'precision': multi_label['overall']['precision'],
                    'recall': multi_label['overall']['recall'],
                    'f1': multi_label['overall']['f1']
                }
            }
            
            # Single-label performance comparison
            single_label_comparison = {
                'per_book': {
                    'accuracy': per_book['single_label']['accuracy'],
                    'precision': per_book['single_label']['precision'],
                    'recall': per_book['single_label']['recall'],
                    'f1': per_book['single_label']['f1']
                },
                'multi_label': {
                    'accuracy': multi_label['single_label']['accuracy'],
                    'precision': multi_label['single_label']['precision'],
                    'recall': multi_label['single_label']['recall'],
                    'f1': multi_label['single_label']['f1']
                }
            }
            
            # Multi-label performance comparison
            multi_label_performance_comparison = {
                'per_book': {
                    'accuracy': per_book['multi_label']['accuracy'],
                    'precision': per_book['multi_label']['precision'],
                    'recall': per_book['multi_label']['recall'],
                    'f1': per_book['multi_label']['f1']
                },
                'multi_label': {
                    'accuracy': multi_label['multi_label']['accuracy'],
                    'precision': multi_label['multi_label']['precision'],
                    'recall': multi_label['multi_label']['recall'],
                    'f1': multi_label['multi_label']['f1']
                }
            }
            
            # Calculate differences
            overall_diff = {
                'accuracy': multi_label['overall']['accuracy'] - per_book['overall']['accuracy'],
                'precision': multi_label['overall']['precision'] - per_book['overall']['precision'],
                'recall': multi_label['overall']['recall'] - per_book['overall']['recall'],
                'f1': multi_label['overall']['f1'] - per_book['overall']['f1']
            }
            
            single_label_diff = {
                'accuracy': multi_label['single_label']['accuracy'] - per_book['single_label']['accuracy'],
                'precision': multi_label['single_label']['precision'] - per_book['single_label']['precision'],
                'recall': multi_label['single_label']['recall'] - per_book['single_label']['recall'],
                'f1': multi_label['single_label']['f1'] - per_book['single_label']['f1']
            }
            
            multi_label_diff = {
                'accuracy': multi_label['multi_label']['accuracy'] - per_book['multi_label']['accuracy'],
                'precision': multi_label['multi_label']['precision'] - per_book['multi_label']['precision'],
                'recall': multi_label['multi_label']['recall'] - per_book['multi_label']['recall'],
                'f1': multi_label['multi_label']['f1'] - per_book['multi_label']['f1']
            }
            
            self.comparison_data[book] = {
                'overall': overall_comparison,
                'single_label': single_label_comparison,
                'multi_label_performance': multi_label_performance_comparison,
                'differences': {
                    'overall': overall_diff,
                    'single_label': single_label_diff,
                    'multi_label': multi_label_diff
                }
            }
        
        logger.info("Comparison metrics calculated")
        
    def create_comparison_plots(self):
        """Create comprehensive comparison plots."""
        logger.info("Creating comparison plots...")
        
        books = list(self.per_book_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Multi-Label vs Per-Book Approach Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Prepare data for plotting
            per_book_values = []
            multi_label_values = []
            
            for book in books:
                per_book_values.append(self.comparison_data[book]['overall']['per_book'][metric])
                multi_label_values.append(self.comparison_data[book]['overall']['multi_label'][metric])
            
            # Create bar plot
            x = np.arange(len(books))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, per_book_values, width, label='Per-Book Approach', 
                          color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, multi_label_values, width, label='Multi-Label Approach', 
                          color='#4ECDC4', alpha=0.8)
            
            ax.set_xlabel('Books')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([book.split()[0] for book in books], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('approach_comparison_overall.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create single-label vs multi-label performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Single-Label vs Multi-Label Performance by Approach', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Prepare data
            per_book_single = []
            per_book_multi = []
            multi_label_single = []
            multi_label_multi = []
            
            for book in books:
                per_book_single.append(self.comparison_data[book]['single_label']['per_book'][metric])
                per_book_multi.append(self.comparison_data[book]['multi_label_performance']['per_book'][metric])
                multi_label_single.append(self.comparison_data[book]['single_label']['multi_label'][metric])
                multi_label_multi.append(self.comparison_data[book]['multi_label_performance']['multi_label'][metric])
            
            # Create grouped bar plot
            x = np.arange(len(books))
            width = 0.2
            
            ax.bar(x - 1.5*width, per_book_single, width, label='Per-Book Single', color='#FF6B6B', alpha=0.8)
            ax.bar(x - 0.5*width, per_book_multi, width, label='Per-Book Multi', color='#FF8E8E', alpha=0.8)
            ax.bar(x + 0.5*width, multi_label_single, width, label='Multi-Label Single', color='#4ECDC4', alpha=0.8)
            ax.bar(x + 1.5*width, multi_label_multi, width, label='Multi-Label Multi', color='#6EDDDD', alpha=0.8)
            
            ax.set_xlabel('Books')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} - Single vs Multi-Label Performance')
            ax.set_xticks(x)
            ax.set_xticklabels([book.split()[0] for book in books], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('approach_comparison_single_vs_multi.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comparison plots created")
        
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        books = list(self.per_book_results.keys())
        
        report = "# Multi-Label vs Per-Book Approach Comparison Report\n\n"
        
        report += "## Executive Summary\n\n"
        report += "This report compares two different approaches for book classification:\n"
        report += "1. **Per-Book Approach**: Individual binary classifiers for each book\n"
        report += "2. **Multi-Label Approach**: Single unified model for all books\n\n"
        
        # Overall performance comparison
        report += "## Overall Performance Comparison\n\n"
        report += "| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |\n"
        report += "|------|-------------|----------------|-----------|--------|\n"
        
        for book in books:
            per_book_f1 = self.comparison_data[book]['overall']['per_book']['f1']
            multi_label_f1 = self.comparison_data[book]['overall']['multi_label']['f1']
            diff = multi_label_f1 - per_book_f1
            winner = "Multi-Label" if diff > 0 else "Per-Book"
            report += f"| {book} | {per_book_f1:.3f} | {multi_label_f1:.3f} | {diff:+.3f} | {winner} |\n"
        
        # Single-label performance comparison
        report += "\n## Single-Label Performance Comparison\n\n"
        report += "| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |\n"
        report += "|------|-------------|----------------|-----------|--------|\n"
        
        for book in books:
            per_book_f1 = self.comparison_data[book]['single_label']['per_book']['f1']
            multi_label_f1 = self.comparison_data[book]['single_label']['multi_label']['f1']
            diff = multi_label_f1 - per_book_f1
            winner = "Multi-Label" if diff > 0 else "Per-Book"
            report += f"| {book} | {per_book_f1:.3f} | {multi_label_f1:.3f} | {diff:+.3f} | {winner} |\n"
        
        # Multi-label performance comparison
        report += "\n## Multi-Label Performance Comparison\n\n"
        report += "| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |\n"
        report += "|------|-------------|----------------|-----------|--------|\n"
        
        for book in books:
            per_book_f1 = self.comparison_data[book]['multi_label_performance']['per_book']['f1']
            multi_label_f1 = self.comparison_data[book]['multi_label_performance']['multi_label']['f1']
            diff = multi_label_f1 - per_book_f1
            winner = "Multi-Label" if diff > 0 else "Per-Book"
            report += f"| {book} | {per_book_f1:.3f} | {multi_label_f1:.3f} | {diff:+.3f} | {winner} |\n"
        
        # Detailed analysis
        report += "\n## Detailed Analysis\n\n"
        
        # Calculate average performance
        avg_per_book_overall = np.mean([self.comparison_data[book]['overall']['per_book']['f1'] for book in books])
        avg_multi_label_overall = np.mean([self.comparison_data[book]['overall']['multi_label']['f1'] for book in books])
        
        avg_per_book_single = np.mean([self.comparison_data[book]['single_label']['per_book']['f1'] for book in books])
        avg_multi_label_single = np.mean([self.comparison_data[book]['single_label']['multi_label']['f1'] for book in books])
        
        avg_per_book_multi = np.mean([self.comparison_data[book]['multi_label_performance']['per_book']['f1'] for book in books])
        avg_multi_label_multi = np.mean([self.comparison_data[book]['multi_label_performance']['multi_label']['f1'] for book in books])
        
        report += "### Average Performance\n\n"
        report += f"- **Overall Performance**:\n"
        report += f"  - Per-Book Approach: {avg_per_book_overall:.3f}\n"
        report += f"  - Multi-Label Approach: {avg_multi_label_overall:.3f}\n"
        report += f"  - Difference: {avg_multi_label_overall - avg_per_book_overall:+.3f}\n\n"
        
        report += f"- **Single-Label Performance**:\n"
        report += f"  - Per-Book Approach: {avg_per_book_single:.3f}\n"
        report += f"  - Multi-Label Approach: {avg_multi_label_single:.3f}\n"
        report += f"  - Difference: {avg_multi_label_single - avg_per_book_single:+.3f}\n\n"
        
        report += f"- **Multi-Label Performance**:\n"
        report += f"  - Per-Book Approach: {avg_per_book_multi:.3f}\n"
        report += f"  - Multi-Label Approach: {avg_multi_label_multi:.3f}\n"
        report += f"  - Difference: {avg_multi_label_multi - avg_per_book_multi:+.3f}\n\n"
        
        # Performance patterns
        report += "### Performance Patterns\n\n"
        
        # Count winners
        overall_winners = sum(1 for book in books if self.comparison_data[book]['overall']['multi_label']['f1'] > self.comparison_data[book]['overall']['per_book']['f1'])
        single_winners = sum(1 for book in books if self.comparison_data[book]['single_label']['multi_label']['f1'] > self.comparison_data[book]['single_label']['per_book']['f1'])
        multi_winners = sum(1 for book in books if self.comparison_data[book]['multi_label_performance']['multi_label']['f1'] > self.comparison_data[book]['multi_label_performance']['per_book']['f1'])
        
        report += f"- **Overall Winners**: Multi-Label wins in {overall_winners}/{len(books)} books\n"
        report += f"- **Single-Label Winners**: Multi-Label wins in {single_winners}/{len(books)} books\n"
        report += f"- **Multi-Label Winners**: Multi-Label wins in {multi_winners}/{len(books)} books\n\n"
        
        # Book-specific analysis
        report += "### Book-Specific Analysis\n\n"
        
        for book in books:
            report += f"#### {book}\n\n"
            
            overall_diff = self.comparison_data[book]['differences']['overall']['f1']
            single_diff = self.comparison_data[book]['differences']['single_label']['f1']
            multi_diff = self.comparison_data[book]['differences']['multi_label']['f1']
            
            report += f"- **Overall**: Multi-Label {'wins' if overall_diff > 0 else 'loses'} by {abs(overall_diff):.3f} F1 points\n"
            report += f"- **Single-Label**: Multi-Label {'wins' if single_diff > 0 else 'loses'} by {abs(single_diff):.3f} F1 points\n"
            report += f"- **Multi-Label**: Multi-Label {'wins' if multi_diff > 0 else 'loses'} by {abs(multi_diff):.3f} F1 points\n\n"
            
            # Interpretation
            if overall_diff > 0:
                report += f"**Interpretation**: The unified multi-label model performs better for {book}, suggesting that shared representations help identify this book's distinctive features.\n\n"
            else:
                report += f"**Interpretation**: The specialized per-book model performs better for {book}, suggesting that focused training on this specific book's characteristics is more effective.\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if avg_multi_label_overall > avg_per_book_overall:
            report += "### Primary Recommendation: Multi-Label Approach\n\n"
            report += "The multi-label approach shows better overall performance and should be preferred for:\n"
            report += "- **Unified Processing**: Single model handles all books\n"
            report += "- **Shared Representations**: Leverages common features across books\n"
            report += "- **Easier Deployment**: One model to maintain and deploy\n"
            report += "- **Better Multi-Label Performance**: Excels at identifying books in mixed contexts\n\n"
        else:
            report += "### Primary Recommendation: Per-Book Approach\n\n"
            report += "The per-book approach shows better overall performance and should be preferred for:\n"
            report += "- **Specialized Performance**: Each model optimized for specific book\n"
            report += "- **Better Single-Label Performance**: Excels at identifying individual books\n"
            report += "- **Modular Design**: Independent models can be updated separately\n"
            report += "- **Interpretability**: Clear which model is responsible for each book\n\n"
        
        report += "### Hybrid Approach Consideration\n\n"
        report += "Consider a hybrid approach based on use case:\n"
        report += "- **Multi-Label Model**: For scenarios with mixed book content\n"
        report += "- **Per-Book Models**: For scenarios with single book identification\n"
        report += "- **Ensemble**: Combine both approaches for maximum accuracy\n\n"
        
        # Technical considerations
        report += "## Technical Considerations\n\n"
        report += "### Multi-Label Approach\n"
        report += "- **Pros**: Unified model, shared representations, easier deployment\n"
        report += "- **Cons**: Complex training, potential for interference between books\n\n"
        
        report += "### Per-Book Approach\n"
        report += "- **Pros**: Specialized models, independent optimization, clear interpretability\n"
        report += "- **Cons**: Multiple models to maintain, no shared learning\n\n"
        
        # Save report
        with open('approach_comparison_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Comparison report generated")
        
    def run_comparison(self):
        """Run the complete comparison analysis."""
        logger.info("Starting approach comparison analysis...")
        
        self.load_results()
        self.calculate_comparison_metrics()
        self.create_comparison_plots()
        self.generate_comparison_report()
        
        logger.info("Comparison analysis completed!")

def main():
    """Main comparison function."""
    comparator = ApproachComparator()
    comparator.run_comparison()

if __name__ == "__main__":
    main() 