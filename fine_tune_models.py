#!/usr/bin/env python3
"""
Fine-tuning script for improving the trained models.
Allows for additional training epochs, hyperparameter tuning, and model optimization.
"""

import torch
import json
import yaml
import argparse
import sys
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
from models.semantic_embedding_model import SemanticEmbeddingModel
from models.multi_label_classifier import SemanticMultiLabelModel
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_model(model_path: str, device: str = "mps"):
    """Load existing trained model for fine-tuning."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get parameters from checkpoint
    num_books = len(checkpoint['book_to_id'])
    embedding_dim = 384  # Default embedding dimension for all-MiniLM-L6-v2
    hidden_size = 256    # Default hidden size
    num_layers = 2       # Default number of layers
    dropout = 0.2        # Default dropout
    
    # Create the complete model
    semantic_model = SemanticEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        embedding_dim=embedding_dim
    )
    
    complete_model = SemanticMultiLabelModel(
        semantic_model=semantic_model,
        num_books=num_books,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    complete_model.load_state_dict(checkpoint['model_state_dict'])
    complete_model.to(device)
    
    return complete_model, checkpoint['book_to_id']

def fine_tune_model(model, train_sentences, train_labels, book_to_id, 
                   val_sentences, val_labels, device, 
                   learning_rate=1e-4, epochs=5, batch_size=16):
    """Fine-tune the model with additional training."""
    logger.info(f"Starting fine-tuning with lr={learning_rate}, epochs={epochs}")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training
        for i in tqdm(range(0, len(train_sentences), batch_size), 
                     desc=f"Epoch {epoch+1}/{epochs}"):
            batch_sentences = train_sentences[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Create targets
            targets = torch.zeros(len(batch_sentences), len(book_to_id))
            for j, label in enumerate(batch_labels):
                if label in book_to_id:
                    targets[j, book_to_id[label]] = 1
            
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(batch_sentences)
            book_scores = outputs['book_scores']
            
            # Calculate loss
            loss = criterion(book_scores, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_sentences) // batch_size + 1)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_sentences), batch_size):
                batch_sentences = val_sentences[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size]
                
                # Create targets
                targets = torch.zeros(len(batch_sentences), len(book_to_id))
                for j, label in enumerate(batch_labels):
                    if label in book_to_id:
                        targets[j, book_to_id[label]] = 1
                
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(batch_sentences)
                book_scores = outputs['book_scores']
                
                # Calculate loss
                loss = criterion(book_scores, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = (book_scores > 0.5).float()
                correct += (predictions == targets).sum().item()
                total += targets.numel()
        
        avg_val_loss = val_loss / (len(val_sentences) // batch_size + 1)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return model, train_losses, val_losses

def evaluate_fine_tuned_model(model, test_sentences, test_labels, book_to_id, device):
    """Evaluate the fine-tuned model."""
    logger.info("Evaluating fine-tuned model...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sentence, label in tqdm(zip(test_sentences, test_labels), total=len(test_sentences)):
            # Get predictions
            outputs = model([sentence])
            book_scores = outputs['book_scores']
            
            # Convert scores to predictions (threshold-based)
            predictions = (book_scores > 0.5).float()
            
            # Convert label to one-hot encoding
            label_one_hot = torch.zeros(len(book_to_id))
            if label in book_to_id:
                label_one_hot[book_to_id[label]] = 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(label_one_hot.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels.flatten(), all_predictions.flatten(), average='binary', zero_division=0
    )
    
    logger.info(f"Fine-tuned model accuracy: {accuracy:.3f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1-Score: {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_fine_tuned_model(model, book_to_id, save_path: str):
    """Save the fine-tuned model."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'book_to_id': book_to_id,
        'fine_tuned': True
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Fine-tuned model saved to {save_path}")

def fine_tune_all_models(device="mps", config_path="configs/config.yaml", learning_rate=1e-4, epochs=5, batch_size=16):
    """Fine-tune all trained models."""
    logger.info(f"Using device: {device}")
    
    # Load data
    dataset = load_from_disk("data/processed_dataset")
    train_sentences = dataset['train']['sentence']
    train_labels = dataset['train']['book_id']
    val_sentences = dataset['validation']['sentence']
    val_labels = dataset['validation']['book_id']
    test_sentences = dataset['test']['sentence']
    test_labels = dataset['test']['book_id']
    
    logger.info(f"Training data: {len(train_sentences)} sentences")
    logger.info(f"Validation data: {len(val_sentences)} sentences")
    logger.info(f"Test data: {len(test_sentences)} sentences")
    
    # Load existing model
    model_path = "experiments/multi_label_classifier/multi_label_classifier.pt"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return False
    
    logger.info("Loading existing model...")
    model, book_to_id = load_existing_model(model_path, device)
    
    # Fine-tune the model
    logger.info("Starting fine-tuning...")
    fine_tuned_model, train_losses, val_losses = fine_tune_model(
        model, train_sentences, train_labels, book_to_id,
        val_sentences, val_labels, device,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate fine-tuned model
    logger.info("Evaluating fine-tuned model...")
    results = evaluate_fine_tuned_model(
        fine_tuned_model, test_sentences, test_labels, book_to_id, device
    )
    
    # Save fine-tuned model
    save_path = "experiments/fine_tuned_models/fine_tuned_model.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_fine_tuned_model(fine_tuned_model, book_to_id, save_path)
    
    # Create visualization of training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fine-tuning Progress')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = Path("experiments/fine_tuned_models/fine_tuning_progress.png")
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    # Save results
    results_path = Path("experiments/fine_tuned_models/fine_tuning_results.json")
    results['train_losses'] = train_losses
    results['val_losses'] = val_losses
    results['hyperparameters'] = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=== FINE-TUNING COMPLETED ===")
    logger.info(f"Device used: {device}")
    logger.info(f"Fine-tuned model saved to: {save_path}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Training progress plot saved to: {plot_path}")
    logger.info(f"Final accuracy: {results['accuracy']:.3f}")
    logger.info(f"Final F1-score: {results['f1']:.3f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Fine-tune trained models")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--model-path", type=str, 
                       default="experiments/multi_label_classifier/multi_label_classifier.pt",
                       help="Path to existing model")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-path", type=str, 
                       default="experiments/fine_tuned_model.pt", help="Path to save fine-tuned model")
    
    args = parser.parse_args()
    
    success = fine_tune_all_models(
        device=args.device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 