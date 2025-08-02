"""
Hyperparameter Optimization for Multi-Label Classification
Optimizes Random Forest, Logistic Regression, and SVM models.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimizer for multi-label classification models."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        """
        Initialize the optimizer.
        
        Args:
            data_path: Path to the augmented dataset
        """
        self.data_path = data_path
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.optimization_results = {}
        
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
        
        # Split the data (train/validation for optimization)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        
        logger.info("Data preparation completed!")
        
    def optimize_random_forest(self):
        """Optimize Random Forest hyperparameters."""
        logger.info("Optimizing Random Forest hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [None, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        model = MultiOutputClassifier(base_rf)
        
        # Create scorer for multi-label
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scorer, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Evaluate on validation set
        y_pred = grid_search.predict(self.X_val)
        val_f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
        
        results = {
            'best_params': best_params,
            'cv_score': best_score,
            'validation_f1': val_f1,
            'best_model': grid_search.best_estimator_
        }
        
        self.optimization_results['random_forest'] = results
        
        logger.info(f"Random Forest optimization completed!")
        logger.info(f"Best CV F1 Score: {best_score:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def optimize_logistic_regression(self):
        """Optimize Logistic Regression hyperparameters."""
        logger.info("Optimizing Logistic Regression hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0, 100.0],
            'estimator__penalty': ['l1', 'l2'],
            'estimator__solver': ['liblinear', 'saga'],
            'estimator__max_iter': [1000, 2000]
        }
        
        # Create base model
        base_lr = LogisticRegression(random_state=42, n_jobs=-1)
        model = MultiOutputClassifier(base_lr)
        
        # Create scorer for multi-label
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scorer, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Evaluate on validation set
        y_pred = grid_search.predict(self.X_val)
        val_f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
        
        results = {
            'best_params': best_params,
            'cv_score': best_score,
            'validation_f1': val_f1,
            'best_model': grid_search.best_estimator_
        }
        
        self.optimization_results['logistic_regression'] = results
        
        logger.info(f"Logistic Regression optimization completed!")
        logger.info(f"Best CV F1 Score: {best_score:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def optimize_svm(self):
        """Optimize SVM hyperparameters."""
        logger.info("Optimizing SVM hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0, 100.0],
            'estimator__kernel': ['linear', 'rbf', 'poly'],
            'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        # Create base model
        base_svm = SVC(probability=True, random_state=42)
        model = MultiOutputClassifier(base_svm)
        
        # Create scorer for multi-label
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scorer, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Evaluate on validation set
        y_pred = grid_search.predict(self.X_val)
        val_f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
        
        results = {
            'best_params': best_params,
            'cv_score': best_score,
            'validation_f1': val_f1,
            'best_model': grid_search.best_estimator_
        }
        
        self.optimization_results['svm'] = results
        
        logger.info(f"SVM optimization completed!")
        logger.info(f"Best CV F1 Score: {best_score:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def optimize_all_models(self):
        """Optimize all three model types."""
        logger.info("Starting hyperparameter optimization for all models...")
        
        # Optimize each model
        self.optimize_random_forest()
        self.optimize_logistic_regression()
        self.optimize_svm()
        
        # Save results
        self.save_optimization_results()
        
        # Find best model
        self.find_best_model()
        
    def save_optimization_results(self):
        """Save optimization results."""
        
        # Create results directory
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Prepare results for saving
        save_results = {}
        for model_name, results in self.optimization_results.items():
            save_results[model_name] = {
                'best_params': results['best_params'],
                'cv_score': results['cv_score'],
                'validation_f1': results['validation_f1']
            }
        
        # Save results
        results_path = results_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save best models
        models_dir = results_dir / "best_models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, results in self.optimization_results.items():
            model_path = models_dir / f"{model_name}_best.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(results['best_model'], f)
        
        logger.info(f"Optimization results saved to: {results_path}")
        logger.info(f"Best models saved to: {models_dir}")
        
    def find_best_model(self):
        """Find the best performing model."""
        best_model = None
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.optimization_results.items():
            if results['validation_f1'] > best_score:
                best_score = results['validation_f1']
                best_model = results['best_model']
                best_model_name = model_name
        
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"Validation F1 Score: {best_score:.4f}")
        logger.info(f"{'='*50}")
        
        # Save best model info (only serializable data)
        serializable_results = {}
        for model_name, results in self.optimization_results.items():
            serializable_results[model_name] = {
                'best_params': results['best_params'],
                'cv_score': results['cv_score'],
                'validation_f1': results['validation_f1']
            }
        
        best_model_info = {
            'best_model': best_model_name,
            'best_score': best_score,
            'all_results': serializable_results
        }
        
        best_model_path = Path("optimization_results") / "best_model_info.json"
        with open(best_model_path, 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        return best_model_name, best_model, best_score

def main():
    """Main optimization function."""
    logger.info("Starting hyperparameter optimization...")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer()
    
    # Load and prepare data
    optimizer.load_data()
    
    # Optimize all models
    optimizer.optimize_all_models()
    
    logger.info("Hyperparameter optimization completed!")

if __name__ == "__main__":
    main() 