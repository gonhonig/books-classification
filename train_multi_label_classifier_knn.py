#!/usr/bin/env python3
"""
Step 5: Train Multi-label Classifier with KNN Features (Balanced Dataset)
Train multi-label classifier using KNN features extracted from fine-tuned semantic embedding model.
"""

import torch
import json
import logging
import argparse
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLabelClassifierKNN:
    """Multi-label classifier using KNN features."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the classifier."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.output_dir = Path("experiments/multi_label_classifier_knn")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata (use balanced dataset)
        metadata_file = self.data_dir / "metadata_balanced.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Fallback to original dataset
            with open(self.data_dir / "metadata.json", 'r') as f:
                self.metadata = json.load(f)
        
        # Load KNN features
        self._load_knn_features()
        
    def _load_knn_features(self):
        """Load KNN features from CSV."""
        features_file = self.data_dir / "features_knn" / "augmented_dataset.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(
                f"KNN features not found: {features_file}\n"
                "Please run 'python extract_features_knn.py' first to extract features."
            )
        
        self.df = pd.read_csv(features_file)
        logger.info(f"Loaded KNN features: {self.df.shape}")
        
        # Extract feature columns and labels
        # Only use numeric columns for features
        self.feature_columns = [col for col in self.df.columns if col.startswith(('similarity_', 'belongs_to_', 'knn_')) and col not in ['knn_best_book']]
        self.label_columns = [col for col in self.df.columns if col.startswith('belongs_to_')]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Label columns: {len(self.label_columns)}")
        
    def prepare_data(self):
        """Prepare training and test data."""
        # Split data based on the original dataset splits
        from datasets import load_from_disk
        
        # Load balanced dataset
        dataset_path = self.data_dir / "processed_dataset_balanced"
        if not dataset_path.exists():
            dataset_path = self.data_dir / "processed_dataset"
        
        dataset = load_from_disk(str(dataset_path))
        
        # Get train/val/test indices
        train_indices = list(range(len(dataset['train'])))
        val_indices = list(range(len(dataset['train']), len(dataset['train']) + len(dataset['validation'])))
        test_indices = list(range(len(dataset['train']) + len(dataset['validation']), len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])))
        
        # Split KNN features accordingly
        self.X_train = self.df.iloc[train_indices][self.feature_columns].values
        self.y_train = self.df.iloc[train_indices][self.label_columns].values
        
        self.X_val = self.df.iloc[val_indices][self.feature_columns].values
        self.y_val = self.df.iloc[val_indices][self.label_columns].values
        
        self.X_test = self.df.iloc[test_indices][self.feature_columns].values
        self.y_test = self.df.iloc[test_indices][self.label_columns].values
        
        logger.info(f"Training data: {self.X_train.shape}")
        logger.info(f"Validation data: {self.X_val.shape}")
        logger.info(f"Test data: {self.X_test.shape}")
        
    def train_model(self):
        """Train the multi-label classifier."""
        # Use Random Forest as specified in config
        model_config = self.config['models']['multi_label_classifier']
        
        self.model = RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training multi-label classifier...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training completed!")
        
    def evaluate_model(self, X, y, split_name: str):
        """Evaluate the model on given data."""
        # Predict
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        hamming = hamming_loss(y, y_pred)
        
        # Calculate per-book accuracy
        book_metrics = {}
        for i, book_name in enumerate(self.metadata['books']):
            book_accuracy = accuracy_score(y[:, i], y_pred[:, i])
            book_metrics[book_name] = book_accuracy
        
        # Calculate average predictions per sentence
        avg_predictions = np.mean(np.sum(y_pred, axis=1))
        
        results = {
            'overall_accuracy': accuracy,
            'hamming_loss': hamming,
            'book_metrics': book_metrics,
            'avg_predictions_per_sentence': avg_predictions,
            'predictions': y_pred,
            'true_labels': y
        }
        
        logger.info(f"\n=== {split_name.upper()} RESULTS ===")
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info(f"Hamming loss: {hamming:.4f}")
        logger.info(f"Average predictions per sentence: {avg_predictions:.2f}")
        logger.info("Per-book accuracy:")
        for book, acc in book_metrics.items():
            logger.info(f"  {book}: {acc:.4f}")
        
        return results
        
    def save_model(self):
        """Save the trained model."""
        model_path = self.output_dir / "multi_label_classifier_knn.pkl"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'label_columns': self.label_columns,
                'metadata': self.metadata,
                'model_type': 'RandomForestClassifier',
                'config': self.config['models']['multi_label_classifier']
            }, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting multi-label classifier training with KNN features...")
        
        # Prepare data
        self.prepare_data()
        
        # Train model
        self.train_model()
        
        # Evaluate on validation set
        val_results = self.evaluate_model(self.X_val, self.y_val, "validation")
        
        # Evaluate on test set
        test_results = self.evaluate_model(self.X_test, self.y_test, "test")
        
        # Save model
        self.save_model()
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            val_results_serializable = {
                'overall_accuracy': float(val_results['overall_accuracy']),
                'hamming_loss': float(val_results['hamming_loss']),
                'book_metrics': {k: float(v) for k, v in val_results['book_metrics'].items()},
                'avg_predictions_per_sentence': float(val_results['avg_predictions_per_sentence'])
            }
            test_results_serializable = {
                'overall_accuracy': float(test_results['overall_accuracy']),
                'hamming_loss': float(test_results['hamming_loss']),
                'book_metrics': {k: float(v) for k, v in test_results['book_metrics'].items()},
                'avg_predictions_per_sentence': float(test_results['avg_predictions_per_sentence'])
            }
            json.dump({
                'validation_results': val_results_serializable,
                'test_results': test_results_serializable,
                'feature_columns': self.feature_columns,
                'label_columns': self.label_columns
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        return val_results, test_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-label classifier with KNN features")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Train model
    classifier = MultiLabelClassifierKNN(args.config)
    val_results, test_results = classifier.run_training_pipeline()
    
    print(f"\n=== MULTI-LABEL CLASSIFIER TRAINING COMPLETED ===")
    print(f"Model saved to: {classifier.output_dir}")
    print(f"Features used: {len(classifier.feature_columns)}")
    
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Overall accuracy: {val_results['overall_accuracy']:.4f}")
    print(f"Hamming loss: {val_results['hamming_loss']:.4f}")
    print(f"Average predictions per sentence: {val_results['avg_predictions_per_sentence']:.2f}")
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Overall accuracy: {test_results['overall_accuracy']:.4f}")
    print(f"Hamming loss: {test_results['hamming_loss']:.4f}")
    print(f"Average predictions per sentence: {test_results['avg_predictions_per_sentence']:.2f}")

if __name__ == "__main__":
    main() 