#!/usr/bin/env python3
"""
Visualization of Multi-Label Classifier Hyperparameter Optimization Results
Create comprehensive visualizations of the optimization process and results.
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationVisualizer:
    """Visualize hyperparameter optimization results."""
    
    def __init__(self, results_dir: str = "experiments/multi_label_optimization"):
        """Initialize the visualizer."""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("experiments/multi_label_optimization/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """Load optimization results."""
        results_file = self.results_dir / "multi_label_optimization_results.json"
        best_params_file = self.results_dir / "best_hyperparameters.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        with open(best_params_file, 'r') as f:
            best_params = json.load(f)
        
        return {
            'results': results,
            'best_params': best_params
        }
    
    def create_performance_comparison(self):
        """Create performance comparison visualization."""
        logger.info("Creating performance comparison visualization...")
        
        # Extract scores
        model_names = list(self.results['results'].keys())
        scores = [self.results['results'][model]['best_score'] for model in model_names]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Bar chart
        colors = ['#2E8B57', '#4682B4', '#CD853F']  # Green, Blue, Brown
        bars = ax1.bar(model_names, scores, color=colors, alpha=0.7)
        ax1.set_title('Multi-Label Classifier Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1-Weighted Score')
        ax1.set_ylim(0.9, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance difference from best
        best_score = max(scores)
        performance_diff = [(score - best_score) * 100 for score in scores]
        
        bars2 = ax2.bar(model_names, performance_diff, color=colors, alpha=0.7)
        ax2.set_title('Performance Difference from Best Model (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Difference from Best (%)')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, diff in zip(bars2, performance_diff):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{diff:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison saved to {self.output_dir}")
    
    def create_parameter_analysis(self):
        """Create parameter analysis visualizations."""
        logger.info("Creating parameter analysis visualizations...")
        
        best_params = self.results['best_params']
        
        # Create subplots for each model
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimized Hyperparameters Analysis', fontsize=16, fontweight='bold')
        
        # 1. Random Forest parameters
        rf_params = best_params.get('random_forest', {})
        if rf_params:
            ax1 = axes[0, 0]
            param_names = list(rf_params.keys())
            param_values = list(rf_params.values())
            
            # Convert values to strings for display
            param_values_str = [str(v) for v in param_values]
            
            bars = ax1.barh(param_names, range(len(param_names)), color='#2E8B57', alpha=0.7)
            ax1.set_title('Random Forest (Best: 0.9733)', fontweight='bold')
            ax1.set_xlim(0, len(param_names))
            ax1.set_xticks([])
            
            # Add parameter values as text
            for i, (name, value) in enumerate(zip(param_names, param_values_str)):
                ax1.text(0.5, i, f'{name}: {value}', ha='center', va='center', fontweight='bold')
        
        # 2. Logistic Regression parameters
        lr_params = best_params.get('logistic_regression', {})
        if lr_params:
            ax2 = axes[0, 1]
            param_names = list(lr_params.keys())
            param_values = list(lr_params.values())
            param_values_str = [str(v) for v in param_values]
            
            bars = ax2.barh(param_names, range(len(param_names)), color='#4682B4', alpha=0.7)
            ax2.set_title('Logistic Regression (Score: 0.9538)', fontweight='bold')
            ax2.set_xlim(0, len(param_names))
            ax2.set_xticks([])
            
            for i, (name, value) in enumerate(zip(param_names, param_values_str)):
                ax2.text(0.5, i, f'{name}: {value}', ha='center', va='center', fontweight='bold')
        
        # 3. SVM parameters
        svm_params = best_params.get('svm', {})
        if svm_params:
            ax3 = axes[1, 0]
            param_names = list(svm_params.keys())
            param_values = list(svm_params.values())
            param_values_str = [str(v) for v in param_values]
            
            bars = ax3.barh(param_names, range(len(param_names)), color='#CD853F', alpha=0.7)
            ax3.set_title('SVM (Score: 0.9539)', fontweight='bold')
            ax3.set_xlim(0, len(param_names))
            ax3.set_xticks([])
            
            for i, (name, value) in enumerate(zip(param_names, param_values_str)):
                ax3.text(0.5, i, f'{name}: {value}', ha='center', va='center', fontweight='bold')
        
        # 4. Performance summary
        ax4 = axes[1, 1]
        model_names = list(self.results['results'].keys())
        scores = [self.results['results'][model]['best_score'] for model in model_names]
        
        # Create a summary table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [
            ['Model', 'Score', 'Rank'],
            ['Random Forest', f'{scores[0]:.4f}', '1st'],
            ['SVM', f'{scores[2]:.4f}', '2nd'],
            ['Logistic Regression', f'{scores[1]:.4f}', '3rd']
        ]
        
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Color the best model row
        table[(1, 0)].set_facecolor('#2E8B57')
        table[(1, 1)].set_facecolor('#2E8B57')
        table[(1, 2)].set_facecolor('#2E8B57')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Parameter analysis saved to {self.output_dir}")
    
    def create_optimization_insights(self):
        """Create insights visualization."""
        logger.info("Creating optimization insights visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Optimization Insights', fontsize=16, fontweight='bold')
        
        # 1. Model performance distribution
        scores = [self.results['results'][model]['best_score'] for model in self.results['results'].keys()]
        model_names = list(self.results['results'].keys())
        
        ax1.bar(model_names, scores, color=['#2E8B57', '#4682B4', '#CD853F'], alpha=0.7)
        ax1.set_title('Best Performance by Model', fontweight='bold')
        ax1.set_ylabel('F1-Weighted Score')
        ax1.set_ylim(0.95, 0.98)
        
        # Add score labels
        for i, score in enumerate(scores):
            ax1.text(i, score + 0.001, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance gap analysis
        best_score = max(scores)
        gaps = [(best_score - score) * 100 for score in scores]
        
        ax2.bar(model_names, gaps, color=['#2E8B57', '#4682B4', '#CD853F'], alpha=0.7)
        ax2.set_title('Performance Gap from Best (%)', fontweight='bold')
        ax2.set_ylabel('Gap (%)')
        
        # Add gap labels
        for i, gap in enumerate(gaps):
            if gap > 0:
                ax2.text(i, gap + 0.05, f'{gap:.2f}%', ha='center', va='bottom')
        
        # 3. Key parameter insights
        ax3.axis('tight')
        ax3.axis('off')
        
        insights_data = [
            ['Insight', 'Value', 'Impact'],
            ['Best Model', 'Random Forest', 'Highest F1 Score'],
            ['Key RF Param', 'n_estimators=179', 'More trees help'],
            ['RF Depth', 'max_depth=13', 'Moderate depth optimal'],
            ['LR Penalty', 'L1 (Lasso)', 'Feature selection'],
            ['SVM Kernel', 'Poly', 'Non-linear patterns'],
            ['Performance Range', '0.9538-0.9733', '2% difference']
        ]
        
        table = ax3.table(cellText=insights_data[1:], colLabels=insights_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # 4. Recommendations
        ax4.axis('tight')
        ax4.axis('off')
        
        recommendations = [
            'RECOMMENDATIONS',
            '',
            '✓ Use Random Forest as primary model',
            '✓ Keep all models for comparison',
            '✓ Monitor performance over time',
            '✓ Consider ensemble methods',
            '✓ Re-optimize if data changes'
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(recommendations), fontsize=12, 
                verticalalignment='top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimization_insights.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization insights saved to {self.output_dir}")
    
    def create_comprehensive_report(self):
        """Create a comprehensive report visualization."""
        logger.info("Creating comprehensive report...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Multi-Label Classifier Hyperparameter Optimization Report', 
                    fontsize=18, fontweight='bold')
        
        # 1. Performance comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        model_names = list(self.results['results'].keys())
        scores = [self.results['results'][model]['best_score'] for model in model_names]
        colors = ['#2E8B57', '#4682B4', '#CD853F']
        
        bars = ax1.bar(model_names, scores, color=colors, alpha=0.7)
        ax1.set_title('Model Performance', fontweight='bold')
        ax1.set_ylabel('F1-Weighted Score')
        ax1.set_ylim(0.95, 0.98)
        
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Best parameters summary (top right)
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.axis('tight')
        ax2.axis('off')
        
        best_params = self.results['best_params']
        summary_data = [
            ['Model', 'Best Score', 'Key Parameters'],
            ['Random Forest', '0.9733', f"n_estimators=179, max_depth=13"],
            ['Logistic Regression', '0.9538', f"C=4.62, penalty=l1"],
            ['SVM', '0.9539', f"C=1.64, kernel=poly"]
        ]
        
        table = ax2.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Color the best model
        table[(1, 0)].set_facecolor('#2E8B57')
        table[(1, 1)].set_facecolor('#2E8B57')
        table[(1, 2)].set_facecolor('#2E8B57')
        
        # 3. Parameter comparison (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create parameter comparison for Random Forest
        rf_params = best_params.get('random_forest', {})
        if rf_params:
            param_names = list(rf_params.keys())
            param_values = [str(v) for v in rf_params.values()]
            
            y_pos = np.arange(len(param_names))
            bars = ax3.barh(y_pos, [1]*len(param_names), color='#2E8B57', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(param_names)
            ax3.set_title('Random Forest Optimized Parameters', fontweight='bold')
            ax3.set_xlim(0, 1)
            ax3.set_xticks([])
            
            for i, value in enumerate(param_values):
                ax3.text(0.5, i, value, ha='center', va='center', fontweight='bold')
        
        # 4. Recommendations (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        recommendations = [
            'OPTIMIZATION RECOMMENDATIONS',
            '',
            '1. Primary Model: Use Random Forest (0.9733 score) for production',
            '2. Baseline Models: Keep SVM and Logistic Regression for comparison',
            '3. Monitoring: Track performance over time and re-optimize if needed',
            '4. Ensemble: Consider combining all three models for better robustness',
            '5. Feature Engineering: Explore additional features to improve SVM/LR performance'
        ]
        
        ax4.text(0.05, 0.9, '\n'.join(recommendations), fontsize=12, 
                verticalalignment='top', fontweight='bold')
        
        plt.savefig(self.output_dir / "comprehensive_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive report saved to {self.output_dir}")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        logger.info("Creating all optimization visualizations...")
        
        try:
            self.create_performance_comparison()
            self.create_parameter_analysis()
            self.create_optimization_insights()
            self.create_comprehensive_report()
            
            logger.info("All visualizations created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description="Visualize hyperparameter optimization results")
    parser.add_argument("--results-dir", default="experiments/multi_label_optimization",
                       help="Directory containing optimization results")
    
    args = parser.parse_args()
    
    try:
        visualizer = OptimizationVisualizer(args.results_dir)
        visualizer.create_all_visualizations()
        
        print(f"\n=== OPTIMIZATION VISUALIZATIONS COMPLETED ===")
        print(f"Visualizations saved to: {visualizer.output_dir}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 