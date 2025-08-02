"""
Test semantic embeddings approach with a simple Logistic Regression model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_semantic_model():
    """Test semantic embeddings with Logistic Regression."""
    
    logger.info("Loading dataset...")
    df = pd.read_csv('data/semantic_augmented/semantic_augmented_dataset.csv')
    logger.info(f"Dataset shape: {df.shape}")
    
    logger.info("Loading semantic embeddings...")
    embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')['embeddings']
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Verify embeddings match dataset
    logger.info("Verifying embeddings match dataset...")
    if len(df) != len(embeddings):
        logger.error(f"ERROR: Shape mismatch! Dataset: {len(df)}, Embeddings: {len(embeddings)}")
        return None
    
    logger.info("✓ Shapes match!")
    
    # Test a few random embeddings
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(df)), 3)
    
    logger.info("Testing embeddings for 3 random sentences...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    for i, idx in enumerate(sample_indices):
        sentence = df.iloc[idx]['sentence']
        cached_embedding = embeddings[idx]
        fresh_embedding = model.encode([sentence])[0]
        
        similarity = np.dot(cached_embedding, fresh_embedding) / (np.linalg.norm(cached_embedding) * np.linalg.norm(fresh_embedding))
        
        logger.info(f"  Sample {i+1}: Similarity = {similarity:.6f}")
        if similarity > 0.9999:
            logger.info(f"    ✓ Embeddings match!")
        else:
            logger.error(f"    ✗ Embeddings don't match!")
            return None
    
    logger.info("✓ All embeddings verified!")
    
    # Prepare features and labels
    X = embeddings  # Use semantic embeddings as features
    label_cols = [col for col in df.columns if col.startswith('book_')]
    y = df[label_cols].values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Check label distribution
    unique_labels_per_sample = np.sum(y, axis=1)
    logger.info(f"Label distribution:")
    logger.info(f"  Single-label samples: {np.sum(unique_labels_per_sample == 1)}")
    logger.info(f"  Multi-label samples: {np.sum(unique_labels_per_sample > 1)}")
    logger.info(f"  Average labels per sample: {np.mean(unique_labels_per_sample):.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Training Logistic Regression model...")
    
    # Train Logistic Regression
    base_lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    model = MultiOutputClassifier(base_lr)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info("Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Hamming Loss: {hamming:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    # Analyze performance by label count
    test_label_counts = np.sum(y_test, axis=1)
    
    logger.info("\nPerformance by label count:")
    for label_count in sorted(set(test_label_counts)):
        mask = test_label_counts == label_count
        if np.sum(mask) > 0:
            subset_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            subset_f1 = f1_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
            logger.info(f"  {label_count} label(s): {np.sum(mask)} samples, Accuracy: {subset_accuracy:.4f}, F1: {subset_f1:.4f}")
    
    # Check average predictions per sample
    avg_predictions = np.mean(np.sum(y_pred, axis=1))
    logger.info(f"\nAverage predictions per sample: {avg_predictions:.2f}")
    
    # Show some examples
    logger.info("\nSample predictions:")
    for i in range(min(5, len(y_test))):
        true_labels = y_test[i]
        pred_labels = y_pred[i]
        true_count = np.sum(true_labels)
        pred_count = np.sum(pred_labels)
        logger.info(f"  Sample {i+1}: True labels: {true_count}, Predicted: {pred_count}")
    
    return {
        'accuracy': accuracy,
        'hamming_loss': hamming,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_predictions': avg_predictions
    }

if __name__ == "__main__":
    results = test_semantic_model()
    logger.info(f"\nTest completed! F1 Score: {results['f1_score']:.4f}") 