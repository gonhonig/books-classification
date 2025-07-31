#!/usr/bin/env python3
"""
Step 5: Train Multi-label Classifier with KNN Features (CORRECTED)
Train multi-label classifier using KNN features extracted from fine-tuned semantic embedding model.
This version correctly separates features from labels and implements multiple algorithms.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLabelClassifierKNNCorrected:
    """Multi-label classifier using KNN features (corrected version)."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the classifier."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.output_dir = Path("experiments/multi_label_classifier_knn_corrected")
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
        
        # CORRECTED: Separate features from labels
        # Features: similarity scores + KNN metadata (excluding belongs_to columns)
        self.feature_columns = [
            col for col in self.df.columns 
            if col.startswith('similarity_') or 
               col in ['knn_best_score', 'knn_confidence', 'num_books_belongs_to']
        ]
        
        # Labels: belongs_to columns (multi-label targets)
        self.label_columns = [col for col in self.df.columns if col.startswith('belongs_to_')]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Label columns: {len(self.label_columns)}")
        logger.info(f"Features: {self.feature_columns}")
        logger.info(f"Labels: {self.label_columns}")
        
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
        
    def train_models(self):
        """Train multiple models with optimized hyperparameters from config."""
        logger.info("Training multi-label classifiers with optimized parameters from config...")
        
        models = {}
        
        # Get optimized parameters from config
        config_models = self.config.get('models', {})
        
        # 1. Random Forest (optimized parameters from config)
        logger.info("Training Random Forest with optimized parameters...")
        rf_params = config_models.get('random_forest', {})
        rf_model = RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 179),
            max_depth=rf_params.get('max_depth', 13),
            min_samples_split=rf_params.get('min_samples_split', 2),
            min_samples_leaf=rf_params.get('min_samples_leaf', 7),
            max_features=rf_params.get('max_features', None),
            bootstrap=rf_params.get('bootstrap', True),
            random_state=rf_params.get('random_state', 42)
        )
        rf_multi = MultiOutputClassifier(rf_model)
        rf_multi.fit(self.X_train, self.y_train)
        models['random_forest'] = rf_multi
        
        # 2. Logistic Regression (optimized parameters from config)
        logger.info("Training Logistic Regression with optimized parameters...")
        lr_params = config_models.get('logistic_regression', {})
        lr_model = LogisticRegression(
            C=lr_params.get('C', 4.62),
            penalty=lr_params.get('penalty', 'l1'),
            solver=lr_params.get('solver', 'saga'),
            max_iter=lr_params.get('max_iter', 1287),
            random_state=lr_params.get('random_state', 42)
        )
        lr_multi = MultiOutputClassifier(lr_model)
        lr_multi.fit(self.X_train, self.y_train)
        models['logistic_regression'] = lr_multi
        
        # 3. SVM (optimized parameters from config)
        logger.info("Training SVM with optimized parameters...")
        svm_params = config_models.get('svm', {})
        svm_model = SVC(
            C=svm_params.get('C', 1.64),
            kernel=svm_params.get('kernel', 'poly'),
            gamma=svm_params.get('gamma', 'scale'),
            probability=svm_params.get('probability', True),
            random_state=svm_params.get('random_state', 42)
        )
        svm_multi = MultiOutputClassifier(svm_model)
        svm_multi.fit(self.X_train, self.y_train)
        models['svm'] = svm_multi
        
        self.models = models
        logger.info("All models trained successfully with optimized parameters from config!")
        
    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_val = model.predict(self.X_val)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics for each split
            train_metrics = self._calculate_metrics(self.y_train, y_pred_train, "train")
            val_metrics = self._calculate_metrics(self.y_val, y_pred_val, "validation")
            test_metrics = self._calculate_metrics(self.y_test, y_pred_test, "test")
            
            results[model_name] = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics,
                'model': model
            }
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
            logger.info(f"  Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, split_name: str):
        """Calculate comprehensive metrics for multi-label classification."""
        # Overall accuracy (exact match)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Hamming loss
        hamming = hamming_loss(y_true, y_pred)
        
        # Per-label metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-book accuracy
        book_metrics = {}
        for i, book in enumerate(self.metadata['books']):
            book_accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
            book_metrics[book] = book_accuracy
        
        # Average predictions per sentence
        avg_predictions = np.mean(np.sum(y_pred, axis=1))
        
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'book_metrics': book_metrics,
            'avg_predictions_per_sentence': avg_predictions
        }
    
    def save_models(self):
        """Save all trained models and results."""
        logger.info("Saving models and results...")
        
        # Save each model
        for model_name, result in self.results.items():
            model_path = self.output_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'label_columns': self.label_columns,
            'metadata': self.metadata,
            'model_types': list(self.results.keys())
        }
        
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'train': {k: float(v) if isinstance(v, np.number) else v for k, v in result['train'].items()},
                'validation': {k: float(v) if isinstance(v, np.number) else v for k, v in result['validation'].items()},
                'test': {k: float(v) if isinstance(v, np.number) else v for k, v in result['test'].items()}
            }
            # Convert book_metrics
            for split in ['train', 'validation', 'test']:
                serializable_results[model_name][split]['book_metrics'] = {
                    k: float(v) for k, v in serializable_results[model_name][split]['book_metrics'].items()
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Models and results saved to: {self.output_dir}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting multi-label classifier training with KNN features...")
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Save models and results
        self.save_models()
        
        # Find best model
        best_model = None
        best_score = 0
        for model_name, result in results.items():
            test_accuracy = result['test']['accuracy']
            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = model_name
        
        logger.info(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
        
        # Print summary
        print(f"\n=== MULTI-LABEL CLASSIFIER TRAINING COMPLETED ===")
        print(f"Models saved to: {self.output_dir}")
        print(f"Best model: {best_model}")
        print(f"Best accuracy: {best_score:.4f}")
        
        return results

def train_multi_label_classifier_corrected(config_path: str = "configs/config.yaml"):
    """Train multi-label classifier with corrected feature/label separation."""
    classifier = MultiLabelClassifierKNNCorrected(config_path=config_path)
    results = classifier.run_training_pipeline()
    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-label classifier with KNN features (corrected)")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        results = train_multi_label_classifier_corrected(config_path=args.config)
        print("Multi-label classifier training completed successfully!")
    except Exception as e:
        logger.error(f"Multi-label classifier training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 