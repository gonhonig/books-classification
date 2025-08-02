"""
Evaluate Optimized Multi-Label Classification Models
Compares the performance of optimized Random Forest, Logistic Regression, and SVM models.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for optimized multi-label classification models."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        """
        Initialize the evaluator.
        
        Args:
            data_path: Path to the augmented dataset
        """
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.evaluation_results = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the augmented dataset
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Separate features and labels
        # Use sentence text as features (we'll create simple features for now)
        # For now, let's use the original_label and original_book as features
        feature_cols = ['original_label', 'original_book']
        label_cols = [col for col in df.columns if col.startswith('book_')]
        
        # Create simple features
        X = df[feature_cols].copy()
        
        # Convert original_book to numeric features
        book_mapping = {
            'Anna Karenina': 0,
            'Frankenstein': 1, 
            'The Adventures of Alice in Wonderland': 2,
            'Wuthering Heights': 3
        }
        X['original_book_encoded'] = X['original_book'].map(book_mapping)
        X = X[['original_label', 'original_book_encoded']].values
        
        y = df[label_cols].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Data preparation completed!")
        
    def load_optimized_models(self, results_dir: str = "optimization_results"):
        """Load optimized models from the results directory."""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            logger.error(f"Results directory not found: {results_path}")
            return
        
        # Load the optimization results
        results_file = results_path / "optimization_results.json"
        if not results_file.exists():
            logger.error("Optimization results file not found!")
            return
        
        logger.info(f"Loading results from: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Load best models
        models_dir = results_path / "best_models"
        if not models_dir.exists():
            logger.error("Best models directory not found!")
            return
        
        model_files = list(models_dir.glob("*_best.pkl"))
        if not model_files:
            logger.error("No best model files found!")
            return
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_best', '')
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f"Loaded {model_name} model")
        
        return results
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        hamming = hamming_loss(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-book metrics
        book_metrics = {}
        for i in range(self.y_test.shape[1]):
            book_accuracy = accuracy_score(self.y_test[:, i], y_pred[:, i])
            book_precision = precision_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            book_recall = recall_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            book_f1 = f1_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            
            book_metrics[f'book_{i}'] = {
                'accuracy': book_accuracy,
                'precision': book_precision,
                'recall': book_recall,
                'f1_score': book_f1
            }
        
        # Average predictions per sample
        avg_predictions = np.mean(np.sum(y_pred, axis=1))
        
        # Confidence analysis
        if hasattr(model, 'predict_proba'):
            confidence_scores = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
        else:
            avg_confidence = None
        
        results = {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'book_metrics': book_metrics,
            'avg_predictions_per_sample': avg_predictions,
            'avg_confidence': avg_confidence,
            'predictions': y_pred,
            'probabilities': y_pred_proba if hasattr(model, 'predict_proba') else None
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg predictions per sample: {avg_predictions:.2f}")
        if avg_confidence:
            logger.info(f"  Avg confidence: {avg_confidence:.4f}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all loaded models."""
        logger.info("Evaluating all optimized models...")
        
        if not self.models:
            logger.error("No models loaded! Cannot evaluate.")
            return
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, model_name)
        
        # Compare models
        self.compare_models()
        
        # Analyze sentence types
        sentence_analysis, df = self.analyze_sentence_types()
        
        # Get test indices for detailed analysis
        _, test_indices = train_test_split(
            range(len(df)), test_size=0.2, random_state=42
        )
        
        # Evaluate performance on different sentence types
        sentence_type_results = self.evaluate_sentence_types(sentence_analysis, df, test_indices)
        
        # Find classification examples for each model
        classification_examples = {}
        for model_name, model in self.models.items():
            examples = self.find_classification_examples(model_name, model, df, test_indices, N=3)
            classification_examples[model_name] = examples
        
        # Save detailed results
        self.save_detailed_results(sentence_type_results, classification_examples, sentence_analysis)
        
        # Save results
        self.save_evaluation_results()
        
        # Create visualizations
        self.create_visualizations()
        
    def compare_models(self):
        """Compare the performance of all models."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Hamming Loss': results['hamming_loss'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'Avg Predictions': results['avg_predictions_per_sample']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model for each metric
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
        best_hamming = comparison_df.loc[comparison_df['Hamming Loss'].idxmin(), 'Model']
        
        logger.info("\nBest Models by Metric:")
        logger.info(f"  Best Accuracy: {best_accuracy}")
        logger.info(f"  Best F1 Score: {best_f1}")
        logger.info(f"  Best Hamming Loss: {best_hamming}")
        
        logger.info("\nDetailed Comparison:")
        logger.info(comparison_df.to_string(index=False))
        
        # Save comparison
        self.comparison_df = comparison_df
        
    def save_evaluation_results(self):
        """Save evaluation results."""
        
        # Create evaluation directory
        eval_dir = Path("evaluation_results")
        eval_dir.mkdir(exist_ok=True)
        
        # Prepare results for saving
        save_results = {}
        for model_name, results in self.evaluation_results.items():
            save_results[model_name] = {
                'accuracy': results['accuracy'],
                'hamming_loss': results['hamming_loss'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'avg_predictions_per_sample': results['avg_predictions_per_sample'],
                'avg_confidence': results['avg_confidence'],
                'book_metrics': results['book_metrics']
            }
        
        # Save results
        results_path = eval_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save comparison DataFrame
        comparison_path = eval_dir / "model_comparison.csv"
        self.comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        logger.info(f"Model comparison saved to: {comparison_path}")
        
    def save_detailed_results(self, sentence_type_results, classification_examples, sentence_analysis):
        """Save detailed analysis results."""
        
        # Create evaluation directory
        eval_dir = Path("evaluation_results")
        eval_dir.mkdir(exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results for JSON serialization
        sentence_type_results_serializable = convert_numpy_types(sentence_type_results)
        classification_examples_serializable = convert_numpy_types(classification_examples)
        sentence_analysis_serializable = convert_numpy_types(sentence_analysis)
        
        # Save sentence type analysis
        sentence_type_path = eval_dir / "sentence_type_analysis.json"
        with open(sentence_type_path, 'w') as f:
            json.dump(sentence_type_results_serializable, f, indent=2)
        
        # Save classification examples
        examples_path = eval_dir / "classification_examples.json"
        with open(examples_path, 'w') as f:
            json.dump(classification_examples_serializable, f, indent=2)
        
        # Save sentence analysis summary
        analysis_summary = {
            'sentence_types': sentence_analysis_serializable,
            'summary': {
                'total_test_samples': len(self.y_test),
                'unique_sentences': sentence_analysis_serializable.get('unique_sentences', {}).get('count', 0),
                'multi_label_sentences': sentence_analysis_serializable.get('multi_label_sentences', {}).get('count', 0),
                'high_similarity_sentences': sentence_analysis_serializable.get('high_similarity_sentences', {}).get('count', 0),
                'low_similarity_sentences': sentence_analysis_serializable.get('low_similarity_sentences', {}).get('count', 0)
            }
        }
        
        summary_path = eval_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SENTENCE TYPE ANALYSIS")
        logger.info("="*60)
        
        for model_name, results in sentence_type_results.items():
            logger.info(f"\n{model_name.upper()} Performance by Sentence Type:")
            for sentence_type, metrics in results.items():
                logger.info(f"  {sentence_type}:")
                logger.info(f"    Count: {metrics['count']}")
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
                logger.info(f"    Precision: {metrics['precision']:.4f}")
                logger.info(f"    Recall: {metrics['recall']:.4f}")
        
        logger.info(f"\nDetailed results saved to:")
        logger.info(f"  Sentence type analysis: {sentence_type_path}")
        logger.info(f"  Prediction examples: {examples_path}")
        logger.info(f"  Analysis summary: {summary_path}")
        
    def analyze_sentence_types(self):
        """Analyze model performance on different types of sentences."""
        logger.info("Analyzing performance on different sentence types...")
        
        # Load original data to get sentences
        df = pd.read_csv(self.data_path)
        
        # Get test indices
        _, test_indices = train_test_split(
            range(len(df)), test_size=0.2, random_state=42
        )
        test_indices = np.array(test_indices)
        
        # Analyze different sentence types
        sentence_analysis = {}
        
        # 1. Unique sentences (belong to only one book)
        unique_mask = np.sum(self.y_test, axis=1) == 1
        if np.any(unique_mask):
            sentence_analysis['unique_sentences'] = {
                'count': np.sum(unique_mask),
                'indices': test_indices[np.where(unique_mask)[0]]
            }
        
        # 2. Multi-label sentences (belong to multiple books)
        multi_mask = np.sum(self.y_test, axis=1) > 1
        if np.any(multi_mask):
            sentence_analysis['multi_label_sentences'] = {
                'count': np.sum(multi_mask),
                'indices': test_indices[np.where(multi_mask)[0]]
            }
        
        # 3. High similarity sentences (check for sentences with high similarity scores)
        # For now, we'll use sentences that belong to multiple books as high similarity
        high_sim_mask = np.sum(self.y_test, axis=1) >= 2
        if np.any(high_sim_mask):
            sentence_analysis['high_similarity_sentences'] = {
                'count': np.sum(high_sim_mask),
                'indices': test_indices[np.where(high_sim_mask)[0]]
            }
        
        # 4. Low similarity sentences (belong to only one book)
        low_sim_mask = np.sum(self.y_test, axis=1) == 1
        if np.any(low_sim_mask):
            sentence_analysis['low_similarity_sentences'] = {
                'count': np.sum(low_sim_mask),
                'indices': test_indices[np.where(low_sim_mask)[0]]
            }
        
        return sentence_analysis, df
    
    def find_classification_examples(self, model_name: str, model, df, test_indices, N=3):
        """Find examples of correct and wrong classifications by similarity level."""
        logger.info(f"Finding classification examples for {model_name}...")
        
        y_pred = model.predict(self.X_test)
        y_true = self.y_test
        
        # Calculate similarity levels (number of books each sentence belongs to)
        similarity_levels = np.sum(y_true, axis=1)
        
        examples = {
            'correct_classifications': {
                'similarity_4': [],
                'similarity_3': [],
                'similarity_2': [],
                'similarity_1': []
            },
            'wrong_classifications': {
                'similarity_4': [],
                'similarity_3': [],
                'similarity_2': [],
                'similarity_1': []
            }
        }
        
        # Find examples for each similarity level
        for similarity in [4, 3, 2, 1]:
            # Find sentences with this similarity level
            sim_mask = similarity_levels == similarity
            
            if np.any(sim_mask):
                # Get indices for this similarity level
                sim_indices = np.where(sim_mask)[0]
                
                # Check if predictions are correct (exact match)
                correct_mask = np.all(y_true[sim_indices] == y_pred[sim_indices], axis=1)
                wrong_mask = ~correct_mask
                
                # Get correct examples
                correct_indices = sim_indices[correct_mask][:N]
                for idx in correct_indices:
                    original_idx = test_indices[idx]
                    sentence = df.iloc[original_idx]['sentence'][:300] + "..."  # Truncate long sentences
                    true_labels = df.iloc[original_idx][[col for col in df.columns if col.startswith('book_')]].to_dict()
                    predicted_labels = {f"predicted_{col}": int(y_pred[idx, i]) 
                                     for i, col in enumerate([col for col in df.columns if col.startswith('book_')])}
                    
                    examples['correct_classifications'][f'similarity_{similarity}'].append({
                        'sentence': sentence,
                        'true_labels': true_labels,
                        'predicted_labels': predicted_labels,
                        'original_index': int(original_idx),
                        'similarity_level': similarity
                    })
                
                # Get wrong examples
                wrong_indices = sim_indices[wrong_mask][:N]
                for idx in wrong_indices:
                    original_idx = test_indices[idx]
                    sentence = df.iloc[original_idx]['sentence'][:300] + "..."  # Truncate long sentences
                    true_labels = df.iloc[original_idx][[col for col in df.columns if col.startswith('book_')]].to_dict()
                    predicted_labels = {f"predicted_{col}": int(y_pred[idx, i]) 
                                     for i, col in enumerate([col for col in df.columns if col.startswith('book_')])}
                    
                    examples['wrong_classifications'][f'similarity_{similarity}'].append({
                        'sentence': sentence,
                        'true_labels': true_labels,
                        'predicted_labels': predicted_labels,
                        'original_index': int(original_idx),
                        'similarity_level': similarity
                    })
        
        return examples
    
    def evaluate_sentence_types(self, sentence_analysis, df, test_indices):
        """Evaluate model performance on different sentence types."""
        logger.info("Evaluating performance on different sentence types...")
        
        sentence_type_results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            model_results = {}
            for sentence_type, info in sentence_analysis.items():
                if info['count'] > 0:
                    # Get indices for this sentence type
                    type_indices = info['indices']
                    # Map back to test set indices
                    test_type_indices = [i for i, orig_idx in enumerate(test_indices) if orig_idx in type_indices]
                    
                    if test_type_indices:
                        y_true_subset = self.y_test[test_type_indices]
                        y_pred_subset = y_pred[test_type_indices]
                        
                        # Calculate metrics for this sentence type
                        accuracy = accuracy_score(y_true_subset, y_pred_subset)
                        f1 = f1_score(y_true_subset, y_pred_subset, average='weighted', zero_division=0)
                        precision = precision_score(y_true_subset, y_pred_subset, average='weighted', zero_division=0)
                        recall = recall_score(y_true_subset, y_pred_subset, average='weighted', zero_division=0)
                        
                        model_results[sentence_type] = {
                            'count': info['count'],
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall
                        }
            
            sentence_type_results[model_name] = model_results
        
        return sentence_type_results
    
    def create_visualizations(self):
        """Create visualization plots for model comparison."""
        if not hasattr(self, 'comparison_df'):
            logger.error("No comparison data available!")
            return
        
        plots_dir = Path("evaluation_results") / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metric comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            sns.barplot(data=self.comparison_df, x='Model', y=metric, ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Hamming Loss comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.comparison_df, x='Model', y='Hamming Loss')
        plt.title('Hamming Loss Comparison (Lower is Better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'hamming_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Average predictions per sample
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.comparison_df, x='Model', y='Avg Predictions')
        plt.title('Average Predictions per Sample')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'avg_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {plots_dir}")

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load and prepare data
    evaluator.load_data()
    
    # Load optimized models
    evaluator.load_optimized_models()
    
    # Evaluate all models
    evaluator.evaluate_all_models()
    
    # Create visualizations
    evaluator.create_visualizations()
    
    logger.info("Model evaluation completed!")

if __name__ == "__main__":
    main() 