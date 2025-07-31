"""
Multi-Label Classifier Model
Trains multi-label classifiers using KNN features for book classification.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import logging
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiLabelClassifier:
    """Multi-label classifier using KNN features."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize multi-label classifier.
        
        Args:
            model_type: Type of classifier ('random_forest', 'logistic_regression', 'svm')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.label_columns = None
        self.metadata = None
        
    def build_model(self) -> MultiOutputClassifier:
        """Build the multi-label classifier model."""
        if self.model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                min_samples_split=self.kwargs.get('min_samples_split', 2),
                min_samples_leaf=self.kwargs.get('min_samples_leaf', 1),
                random_state=self.kwargs.get('random_state', 42),
                **{k: v for k, v in self.kwargs.items() 
                   if k not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']}
            )
            
        elif self.model_type == "logistic_regression":
            base_model = LogisticRegression(
                random_state=self.kwargs.get('random_state', 42),
                max_iter=self.kwargs.get('max_iter', 1000),
                C=self.kwargs.get('C', 1.0),
                penalty=self.kwargs.get('penalty', 'l2'),
                solver=self.kwargs.get('solver', 'lbfgs'),
                **{k: v for k, v in self.kwargs.items() 
                   if k not in ['random_state', 'max_iter', 'C', 'penalty', 'solver']}
            )
            
        elif self.model_type == "svm":
            base_model = SVC(
                kernel=self.kwargs.get('kernel', 'linear'),
                C=self.kwargs.get('C', 1.0),
                gamma=self.kwargs.get('gamma', 'scale'),
                probability=self.kwargs.get('probability', True),
                random_state=self.kwargs.get('random_state', 42),
                **{k: v for k, v in self.kwargs.items() 
                   if k not in ['kernel', 'C', 'gamma', 'probability', 'random_state']}
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = MultiOutputClassifier(base_model)
        return self.model
    
    def load_data(self, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare training data.
        
        Args:
            data_dir: Directory containing the data
            
        Returns:
            X: Feature matrix
            y: Label matrix
        """
        data_path = Path(data_dir)
        
        # Load metadata
        metadata_file = data_path / "metadata_balanced.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Fallback to original dataset
            with open(data_path / "metadata.json", 'r') as f:
                self.metadata = json.load(f)
        
        # Load KNN features
        features_file = data_path / "features_knn" / "augmented_dataset.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(
                f"KNN features not found: {features_file}\n"
                "Please run 'python extract_features_knn.py' first to extract features."
            )
        
        df = pd.read_csv(features_file)
        logger.info(f"Loaded KNN features: {df.shape}")
        
        # Separate features from labels
        self.feature_columns = [
            col for col in df.columns 
            if col.startswith('similarity_') or 
               col in ['knn_best_score', 'knn_confidence', 'num_books_belongs_to']
        ]
        
        self.label_columns = [col for col in df.columns if col.startswith('belongs_to_')]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Label columns: {len(self.label_columns)}")
        
        # Prepare data
        X = df[self.feature_columns].values
        y = df[self.label_columns].values
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> MultiOutputClassifier:
        """
        Train the multi-label classifier.
        
        Args:
            X: Feature matrix
            y: Label matrix
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_type} multi-label classifier...")
        self.model.fit(X, y)
        logger.info("Training completed!")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        hamming = hamming_loss(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Per-book accuracy
        book_metrics = {}
        if self.metadata and 'books' in self.metadata:
            for i, book in enumerate(self.metadata['books']):
                if i < y.shape[1]:
                    book_accuracy = accuracy_score(y[:, i], y_pred[:, i])
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
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")

def create_multi_label_classifier(model_type: str = "random_forest", **kwargs) -> MultiLabelClassifier:
    """
    Factory function to create a multi-label classifier.
    
    Args:
        model_type: Type of classifier
        **kwargs: Model-specific parameters
        
    Returns:
        MultiLabelClassifier instance
    """
    return MultiLabelClassifier(model_type=model_type, **kwargs) 