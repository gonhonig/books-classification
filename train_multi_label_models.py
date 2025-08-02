"""
Multi-Label Classification Training Script
Trains Random Forest, Logistic Regression, and SVM models with hyperparameter optimization.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiLabelTrainer:
    """Trainer for multi-label classification models."""
    
    def __init__(self, data_path: str = "data/semantic_augmented/semantic_augmented_dataset.csv"):
        """
        Initialize the trainer.
        
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
        self.results = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the augmented dataset
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Load semantic embeddings
        logger.info("Loading semantic embeddings...")
        embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')
        semantic_features = embeddings['embeddings']
        logger.info(f"Loaded semantic embeddings with shape: {semantic_features.shape}")
        
        # Separate labels
        label_cols = [col for col in df.columns if col.startswith('book_')]
        y = df[label_cols].values
        
        # Use semantic embeddings as features
        X = semantic_features
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Number of unique labels per sample: {np.sum(y, axis=1).mean():.2f}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Data preparation completed!")
        
    def train_random_forest(self, **params):
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        rf = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(rf)
        model.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_logistic_regression(self, **params):
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression model...")
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=params.get('max_iter', 1000),
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver=params.get('solver', 'lbfgs'),
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(lr)
        model.fit(self.X_train, self.y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_svm(self, **params):
        """Train SVM model."""
        logger.info("Training SVM model...")
        
        svm = SVC(
            kernel=params.get('kernel', 'linear'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            probability=True,
            random_state=42
        )
        
        model = MultiOutputClassifier(svm)
        model.fit(self.X_train, self.y_train)
        
        self.models['svm'] = model
        return model
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a trained model."""
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(self.X_test)
        
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
            book_metrics[f'book_{i}'] = book_accuracy
        
        results = {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'book_metrics': book_metrics,
            'avg_predictions_per_sample': np.mean(np.sum(y_pred, axis=1))
        }
        
        self.results[model_name] = results
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg predictions per sample: {results['avg_predictions_per_sample']:.2f}")
        
        return results
    
    def save_model(self, model, model_name: str, results: dict):
        """Save trained model and results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        models_dir = Path("trained_models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{model_name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save results
        results_path = models_dir / f"{model_name}_{timestamp}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        
        return model_path, results_path
    
    def train_all_models(self):
        """Train all three model types with default parameters."""
        logger.info("Training all models with default parameters...")
        
        # Train Random Forest
        self.train_random_forest()
        self.evaluate_model(self.models['random_forest'], 'random_forest')
        
        # Train Logistic Regression
        self.train_logistic_regression()
        self.evaluate_model(self.models['logistic_regression'], 'logistic_regression')
        
        # Train SVM
        self.train_svm()
        self.evaluate_model(self.models['svm'], 'svm')
        
        # Save all models and results
        for model_name, model in self.models.items():
            self.save_model(model, model_name, self.results[model_name])
        
        # Save summary
        self.save_summary()
        
    def save_summary(self):
        """Save a summary of all results."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': self.X_train.shape[1],
                'num_labels': self.y_train.shape[1]
            },
            'results': self.results
        }
        
        summary_path = Path("trained_models") / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")

def main():
    """Main training function."""
    logger.info("Starting multi-label classification training...")
    
    # Initialize trainer
    trainer = MultiLabelTrainer()
    
    # Load and prepare data
    trainer.load_data()
    
    # Train all models
    trainer.train_all_models()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 