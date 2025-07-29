"""
Standalone evaluation script for the book classification model.
"""

import os
import sys
import torch
import yaml
import json
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constructive_model import ConstructiveLearningModel
from utils.evaluation import evaluate_model
from utils.data_utils import create_dataloaders, tokenize_dataset
from datasets import load_from_disk

def main():
    """Main evaluation function."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset_path = Path("data/processed_dataset")
    if not dataset_path.exists():
        logger.error("Processed dataset not found. Run data/prepare_data.py first.")
        return
    
    dataset = load_from_disk(str(dataset_path))
    logger.info(f"Loaded dataset with {len(dataset['test'])} test samples")
    
    # Load metadata
    metadata_path = Path("data/metadata.json")
    if not metadata_path.exists():
        logger.error("Metadata not found.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = ConstructiveLearningModel()
    model.to(device)
    
    # Load best checkpoint
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        logger.error("No checkpoints found. Train the model first.")
        return
    
    # Find best checkpoint
    best_checkpoint = None
    best_val_loss = float('inf')
    
    for checkpoint in checkpoints:
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        val_loss = checkpoint_data.get('val_loss', float('inf'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = checkpoint
    
    if best_checkpoint:
        logger.info(f"Loading best checkpoint: {best_checkpoint}")
        checkpoint_data = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
    else:
        logger.error("No valid checkpoint found.")
        return
    
    # Prepare dataloaders
    tokenizer = model.tokenizer
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, 
                                       max_length=config['data']['max_sentence_length'])
    
    test_dataloader = create_dataloaders(
        {'test': tokenized_dataset['test']},
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )['test']
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate_model(model, test_dataloader, device, metadata)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    overall = metrics['overall']
    print(f"Overall Accuracy: {overall['accuracy']:.4f}")
    print(f"Macro F1: {overall['macro_f1']:.4f}")
    print(f"Weighted F1: {overall['weighted_f1']:.4f}")
    
    print("\nPer-class Performance:")
    print("-" * 40)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1-Score: {class_metrics['f1']:.4f}")
        print(f"  Support: {class_metrics['support']}")
        print()
    
    if 'confidence_analysis' in metrics:
        conf = metrics['confidence_analysis']
        print("Confidence Analysis:")
        print("-" * 40)
        print(f"  Mean Confidence: {conf['mean_confidence']:.4f}")
        print(f"  Std Confidence: {conf['std_confidence']:.4f}")
        print(f"  Confidence-Accuracy Correlation: {conf['confidence_accuracy_correlation']:.4f}")
    
    print("\nEvaluation completed!")
    print("Check experiments/results/ for detailed results and visualizations.")

if __name__ == "__main__":
    main() 