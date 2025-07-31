#!/usr/bin/env python3
"""
Runner script for Multi-Label Classifier Hyperparameter Optimization
Run hyperparameter optimization for multi-label classifiers using KNN features.
"""

import logging
import argparse
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if KNN features exist
    features_file = Path("data/features_knn/augmented_dataset.csv")
    if not features_file.exists():
        logger.error(f"KNN features not found: {features_file}")
        logger.error("Please run 'python extract_features_knn.py' first to extract features.")
        return False
    
    # Check if balanced dataset exists
    dataset_path = Path("data/processed_dataset_balanced")
    if not dataset_path.exists():
        logger.error(f"Balanced dataset not found: {dataset_path}")
        logger.error("Please run 'python data/prepare_data_balanced.py' first to prepare the dataset.")
        return False
    
    # Check if metadata exists
    metadata_file = Path("data/metadata_balanced.json")
    if not metadata_file.exists():
        logger.error(f"Balanced metadata not found: {metadata_file}")
        logger.error("Please run 'python data/prepare_data_balanced.py' first to prepare the dataset.")
        return False
    
    logger.info("All prerequisites met!")
    return True

def run_optimization(config_path: str = "configs/config.yaml"):
    """Run the hyperparameter optimization."""
    from hyperparameter_optimization_multi_label import optimize_multi_label_hyperparameters
    
    logger.info("Starting multi-label classifier hyperparameter optimization...")
    
    # Run optimization
    results = optimize_multi_label_hyperparameters(config_path)
    
    # Print summary
    print("\n" + "="*70)
    print("MULTI-LABEL CLASSIFIER HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*70)
    
    best_model = None
    best_score = 0.0
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper()}: ERROR - {result['error']}")
        else:
            score = result['best_score']
            print(f"{model_type.upper()}:")
            print(f"  Best Score: {score:.4f}")
            print(f"  Best Parameters: {result['best_params']}")
            print()
            
            if score > best_score:
                best_score = score
                best_model = model_type
    
    if best_model:
        print(f"BEST MODEL: {best_model.upper()} (Score: {best_score:.4f})")
    
    print(f"\nResults saved to: experiments/multi_label_optimization")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for multi-label classifiers")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip prerequisite checks")
    
    args = parser.parse_args()
    
    try:
        # Check prerequisites
        if not args.skip_checks:
            if not check_prerequisites():
                logger.error("Prerequisites not met. Exiting.")
                sys.exit(1)
        
        # Run optimization
        results = run_optimization(args.config)
        
        logger.info("Hyperparameter optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 