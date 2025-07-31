#!/usr/bin/env python3
"""
Hyperparameter Optimization for Multi-Label Classifier
Optimize hyperparameters for multi-label classification using KNN features.
"""

import torch
import json
import logging
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.samplers import TPESampler
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our model
from models.multi_label_classifier import create_multi_label_classifier

class MultiLabelHyperparameterOptimizer:
    """Hyperparameter optimizer for multi-label classifiers."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the optimizer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.output_dir = Path("experiments/multi_label_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.X, self.y = self._load_data()
        
        # Optimization parameters
        self.n_trials = self.config.get('optimization', {}).get('n_trials', 50)
        self.cv_folds = self.config.get('optimization', {}).get('cv_folds', 5)
        self.timeout = self.config.get('optimization', {}).get('timeout', 3600)  # 1 hour
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for optimization."""
        logger.info("Loading data for hyperparameter optimization...")
        
        # Create a temporary classifier to load data
        temp_classifier = create_multi_label_classifier("random_forest")
        X, y = temp_classifier.load_data(str(self.data_dir))
        
        logger.info(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def objective_random_forest(self, trial: optuna.Trial) -> float:
        """Objective function for Random Forest optimization."""
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        
        # Create and train model
        classifier = create_multi_label_classifier("random_forest", **params)
        classifier.build_model()
        
        # Cross-validation
        scores = cross_val_score(
            classifier.model, 
            self.X, 
            self.y, 
            cv=self.cv_folds, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_logistic_regression(self, trial: optuna.Trial) -> float:
        """Objective function for Logistic Regression optimization."""
        # Define hyperparameter search space
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 500, 2000),
            'random_state': 42
        }
        
        # Create and train model
        classifier = create_multi_label_classifier("logistic_regression", **params)
        classifier.build_model()
        
        # Cross-validation
        scores = cross_val_score(
            classifier.model, 
            self.X, 
            self.y, 
            cv=self.cv_folds, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def objective_svm(self, trial: optuna.Trial) -> float:
        """Objective function for SVM optimization."""
        # Define hyperparameter search space
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True,
            'random_state': 42
        }
        
        # Create and train model
        classifier = create_multi_label_classifier("svm", **params)
        classifier.build_model()
        
        # Cross-validation
        scores = cross_val_score(
            classifier.model, 
            self.X, 
            self.y, 
            cv=self.cv_folds, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize_model(self, model_type: str) -> Tuple[Dict, float]:
        """
        Optimize hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model to optimize ('random_forest', 'logistic_regression', 'svm')
            
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting hyperparameter optimization for {model_type}...")
        
        # Select objective function
        if model_type == "random_forest":
            objective = self.objective_random_forest
        elif model_type == "logistic_regression":
            objective = self.objective_logistic_regression
        elif model_type == "svm":
            objective = self.objective_svm
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best {model_type} parameters: {best_params}")
        logger.info(f"Best {model_type} score: {best_score:.4f}")
        
        return best_params, best_score
    
    def optimize_all_models(self) -> Dict[str, Dict]:
        """
        Optimize hyperparameters for all model types.
        
        Returns:
            Dictionary with best parameters for each model type
        """
        logger.info("Starting hyperparameter optimization for all models...")
        
        results = {}
        model_types = ['random_forest', 'logistic_regression', 'svm']
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Optimizing {model_type.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                best_params, best_score = self.optimize_model(model_type)
                results[model_type] = {
                    'best_params': best_params,
                    'best_score': best_score
                }
            except Exception as e:
                logger.error(f"Error optimizing {model_type}: {e}")
                results[model_type] = {
                    'best_params': {},
                    'best_score': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def save_results(self, results: Dict[str, Dict]):
        """Save optimization results."""
        # Save detailed results
        results_file = self.output_dir / "multi_label_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best parameters for each model
        best_params_file = self.output_dir / "best_hyperparameters.json"
        best_params = {}
        
        for model_type, result in results.items():
            if 'best_params' in result and result['best_params']:
                best_params[model_type] = result['best_params']
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        
        for model_type, result in results.items():
            if 'error' in result:
                print(f"{model_type.upper()}: ERROR - {result['error']}")
            else:
                print(f"{model_type.upper()}:")
                print(f"  Best Score: {result['best_score']:.4f}")
                print(f"  Best Params: {result['best_params']}")
                print()
    
    def run_optimization(self):
        """Run the complete hyperparameter optimization pipeline."""
        logger.info("Starting multi-label classifier hyperparameter optimization...")
        
        # Optimize all models
        results = self.optimize_all_models()
        
        # Save results
        self.save_results(results)
        
        logger.info("Hyperparameter optimization completed!")
        return results

def optimize_multi_label_hyperparameters(config_path: str = "configs/config.yaml") -> Dict[str, Dict]:
    """
    Optimize hyperparameters for multi-label classifiers.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = MultiLabelHyperparameterOptimizer(config_path=config_path)
    return optimizer.run_optimization()

def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for multi-label classifiers")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        results = optimize_multi_label_hyperparameters(args.config)
        
        print(f"\n=== HYPERPARAMETER OPTIMIZATION COMPLETED ===")
        print(f"Results saved to: experiments/multi_label_optimization")
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 