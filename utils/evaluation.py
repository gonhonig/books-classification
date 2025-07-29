"""
Evaluation utilities for book sentence classification model.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def evaluate_model(model, dataloader, device, metadata, save_results: bool = True) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        metadata: Dataset metadata
        save_results: Whether to save results to file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task='classification'
            )
            
            # Get predictions
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, all_probabilities, metadata)
    
    # Create visualizations
    create_evaluation_visualizations(all_predictions, all_labels, all_probabilities, metadata)
    
    # Save results
    if save_results:
        save_evaluation_results(metrics, metadata)
    
    return metrics

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, 
                     probabilities: np.ndarray, metadata: Dict) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Create detailed metrics dictionary
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1)
        },
        'per_class': {}
    }
    
    # Add per-class metrics
    for i in range(len(metadata['id_to_label'])):
        class_name = metadata['id_to_label'][str(i)]
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'accuracy': float(per_class_accuracy[i])
        }
    
    # Add confusion matrix
    metrics['confusion_matrix'] = cm.tolist()
    
    # Add prediction confidence analysis
    confidence_metrics = analyze_prediction_confidence(probabilities, predictions, labels)
    metrics['confidence_analysis'] = confidence_metrics
    
    return metrics

def analyze_prediction_confidence(probabilities: np.ndarray, predictions: np.ndarray, 
                                labels: np.ndarray) -> Dict:
    """Analyze prediction confidence and its relationship with accuracy."""
    
    # Get confidence scores (probability of predicted class)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Calculate accuracy by confidence bins
    confidence_bins = np.linspace(0, 1, 11)  # 10 bins
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(predictions[mask] == labels[mask])
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    # Calculate calibration metrics
    from sklearn.calibration import calibration_curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels == predictions, confidence_scores, n_bins=10
        )
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    except:
        calibration_error = None
    
    return {
        'mean_confidence': float(np.mean(confidence_scores)),
        'std_confidence': float(np.std(confidence_scores)),
        'confidence_bins': confidence_bins.tolist(),
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'calibration_error': calibration_error,
        'confidence_accuracy_correlation': float(np.corrcoef(confidence_scores, predictions == labels)[0, 1])
    }

def create_evaluation_visualizations(predictions: np.ndarray, labels: np.ndarray,
                                   probabilities: np.ndarray, metadata: Dict):
    """Create evaluation visualizations."""
    
    # Create output directory
    output_dir = Path("experiments/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, predictions)
    class_names = [metadata['id_to_label'][str(i)] for i in range(len(metadata['id_to_label']))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class Performance
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [precision, recall, f1]
    metric_names = ['Precision', 'Recall', 'F1-Score']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        axes[i].bar(class_names, metric)
        axes[i].set_title(f'{name} by Class')
        axes[i].set_ylabel(name)
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence Analysis
    confidence_scores = np.max(probabilities, axis=1)
    correct_predictions = predictions == labels
    
    plt.figure(figsize=(12, 5))
    
    # Confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores[correct_predictions], alpha=0.7, label='Correct', bins=20)
    plt.hist(confidence_scores[~correct_predictions], alpha=0.7, label='Incorrect', bins=20)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # Accuracy vs Confidence
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(correct_predictions[mask])
            bin_accuracies.append(bin_accuracy)
            bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
    
    plt.plot(bin_centers, bin_accuracies, 'o-')
    plt.plot([0, 1], [0, 1], '--', alpha=0.5, label='Perfect Calibration')
    plt.xlabel('Confidence Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Error Analysis
    error_indices = np.where(predictions != labels)[0]
    if len(error_indices) > 0:
        error_confidences = confidence_scores[error_indices]
        error_predictions = predictions[error_indices]
        error_labels = labels[error_indices]
        
        plt.figure(figsize=(10, 6))
        plt.hist(error_confidences, bins=20, alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Distribution of Incorrect Predictions')
        plt.tight_layout()
        plt.savefig(output_dir / 'error_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_evaluation_results(metrics: Dict, metadata: Dict):
    """Save evaluation results to file."""
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create summary report
    report = create_evaluation_report(metrics, metadata)
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report)

def create_evaluation_report(metrics: Dict, metadata: Dict) -> str:
    """Create a human-readable evaluation report."""
    
    report = []
    report.append("=" * 60)
    report.append("BOOK SENTENCE CLASSIFICATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall metrics
    overall = metrics['overall']
    report.append("OVERALL PERFORMANCE:")
    report.append(f"  Accuracy: {overall['accuracy']:.4f}")
    report.append(f"  Macro F1: {overall['macro_f1']:.4f}")
    report.append(f"  Weighted F1: {overall['weighted_f1']:.4f}")
    report.append("")
    
    # Per-class performance
    report.append("PER-CLASS PERFORMANCE:")
    report.append("-" * 40)
    
    for class_name, class_metrics in metrics['per_class'].items():
        report.append(f"{class_name}:")
        report.append(f"  Precision: {class_metrics['precision']:.4f}")
        report.append(f"  Recall: {class_metrics['recall']:.4f}")
        report.append(f"  F1-Score: {class_metrics['f1']:.4f}")
        report.append(f"  Support: {class_metrics['support']}")
        report.append("")
    
    # Confidence analysis
    confidence = metrics['confidence_analysis']
    report.append("CONFIDENCE ANALYSIS:")
    report.append("-" * 40)
    report.append(f"  Mean Confidence: {confidence['mean_confidence']:.4f}")
    report.append(f"  Std Confidence: {confidence['std_confidence']:.4f}")
    report.append(f"  Confidence-Accuracy Correlation: {confidence['confidence_accuracy_correlation']:.4f}")
    if confidence['calibration_error'] is not None:
        report.append(f"  Calibration Error: {confidence['calibration_error']:.4f}")
    report.append("")
    
    # Dataset statistics
    report.append("DATASET STATISTICS:")
    report.append("-" * 40)
    report.append(f"  Number of Classes: {metadata['num_classes']}")
    report.append(f"  Test Set Size: {metadata['test_size']}")
    report.append("")
    
    return "\n".join(report)

def cross_validate_model(model_class, dataset, n_folds: int = 5, **kwargs) -> Dict:
    """Perform cross-validation on the model."""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # Get labels for stratification
    labels = dataset['train']['label']
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset['train'], labels)):
        print(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        train_data = dataset['train'].select(train_idx)
        val_data = dataset['train'].select(val_idx)
        
        # Create fold dataset
        fold_dataset = {
            'train': train_data,
            'validation': val_data,
            'test': dataset['test']
        }
        
        # Train and evaluate model
        # This is a simplified version - you'd need to implement the actual training
        fold_metrics = {'fold': fold + 1, 'accuracy': 0.0}  # Placeholder
        fold_results.append(fold_metrics)
    
    # Aggregate results
    accuracies = [result['accuracy'] for result in fold_results]
    
    cv_results = {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies)
    }
    
    return cv_results 