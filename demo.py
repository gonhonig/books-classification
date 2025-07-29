#!/usr/bin/env python3
"""
Demo script for the Book Sentence Classification system.
This script demonstrates the complete pipeline from data preparation to model training and evaluation.
"""

import os
import sys
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_demo():
    """Run the complete demo pipeline."""
    
    logger.info("Starting Book Sentence Classification Demo")
    logger.info("=" * 50)
    
    # Step 1: Data Preparation
    logger.info("Step 1: Preparing data...")
    try:
        from data.prepare_data import main as prepare_data
        prepare_data()
        logger.info("✓ Data preparation completed")
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}")
        return
    
    # Step 2: Model Training
    logger.info("Step 2: Training model...")
    try:
        from models.train import main as train_model
        train_model()
        logger.info("✓ Model training completed")
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        return
    
    # Step 3: Model Evaluation
    logger.info("Step 3: Evaluating model...")
    try:
        from utils.evaluate import main as evaluate_model
        evaluate_model()
        logger.info("✓ Model evaluation completed")
    except Exception as e:
        logger.error(f"✗ Model evaluation failed: {e}")
        return
    
    # Step 4: Show Results
    logger.info("Step 4: Displaying results...")
    show_results()
    
    logger.info("Demo completed successfully!")
    logger.info("Check the following directories for results:")
    logger.info("  - experiments/results/ (evaluation metrics)")
    logger.info("  - experiments/visualizations/ (plots and charts)")
    logger.info("  - experiments/checkpoints/ (model checkpoints)")

def show_results():
    """Display demo results."""
    
    # Load configuration
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Show dataset statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Number of classes: {metadata['num_classes']}")
        print(f"Training samples: {metadata['train_size']}")
        print(f"Validation samples: {metadata['val_size']}")
        print(f"Test samples: {metadata['test_size']}")
        print(f"Total samples: {metadata['total_size']}")
        
        print("\nClass mapping:")
        for class_id, class_name in metadata['id_to_label'].items():
            print(f"  {class_id}: {class_name}")
    
    # Show evaluation results
    results_path = Path("experiments/results/evaluation_metrics.json")
    if results_path.exists():
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        overall = results['overall']
        print(f"Overall Accuracy: {overall['accuracy']:.4f}")
        print(f"Macro F1-Score: {overall['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {overall['weighted_f1']:.4f}")
        
        print("\nPer-class Performance:")
        for class_name, metrics in results['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1']:.4f}")
    
    # Show available visualizations
    viz_dir = Path("experiments/visualizations")
    if viz_dir.exists():
        print("\n" + "="*60)
        print("AVAILABLE VISUALIZATIONS")
        print("="*60)
        
        viz_files = list(viz_dir.glob("*.png"))
        if viz_files:
            for viz_file in viz_files:
                print(f"  - {viz_file.name}")
        else:
            print("  No visualization files found.")

def interactive_demo():
    """Interactive demo for testing individual sentences."""
    
    logger.info("Starting Interactive Demo")
    logger.info("=" * 30)
    
    # Load model and tokenizer
    try:
        from models.constructive_model import ConstructiveLearningModel
        import torch
        
        model = ConstructiveLearningModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load best checkpoint
        checkpoint_dir = Path("experiments/checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        if checkpoints:
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
                checkpoint_data = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info(f"Loaded model from {best_checkpoint}")
            else:
                logger.warning("No valid checkpoint found. Using untrained model.")
        else:
            logger.warning("No checkpoints found. Using untrained model.")
        
        # Load metadata
        metadata_path = Path("data/metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            logger.error("Metadata not found.")
            return
        
        # Interactive testing
        print("\nInteractive Sentence Classification Demo")
        print("Enter sentences to classify (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            sentence = input("\nEnter a sentence: ").strip()
            
            if sentence.lower() == 'quit':
                break
            
            if not sentence:
                continue
            
            # Classify sentence
            try:
                inputs = model.tokenizer(
                    sentence,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        task='classification'
                    )
                    
                    probabilities = torch.softmax(outputs['logits'], dim=-1)
                    prediction = torch.argmax(outputs['logits'], dim=-1)
                    confidence = torch.max(probabilities, dim=-1)[0]
                
                predicted_class = metadata['id_to_label'][str(prediction.item())]
                
                print(f"\nPrediction: {predicted_class}")
                print(f"Confidence: {confidence.item():.4f}")
                
                # Show all probabilities
                print("\nAll class probabilities:")
                for i, prob in enumerate(probabilities[0]):
                    class_name = metadata['id_to_label'][str(i)]
                    print(f"  {class_name}: {prob.item():.4f}")
                    
            except Exception as e:
                print(f"Error classifying sentence: {e}")
        
        print("\nInteractive demo completed!")
        
    except Exception as e:
        logger.error(f"Interactive demo failed: {e}")

def main():
    """Main demo function."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Book Sentence Classification Demo")
    parser.add_argument("--mode", choices=["full", "interactive"], default="full",
                       help="Demo mode: full pipeline or interactive testing")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_demo()
    elif args.mode == "interactive":
        interactive_demo()
    else:
        print("Invalid mode. Use 'full' or 'interactive'.")

if __name__ == "__main__":
    main() 