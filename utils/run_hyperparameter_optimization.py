#!/usr/bin/env python3
"""
Hyperparameter Optimization Runner
Optimizes hyperparameters for both multi-label and per-book neural networks.
"""

import argparse
import logging
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.hyperparameter_optimizer import (
    HyperparameterOptimizer, 
    optimize_multi_label_model, 
    optimize_per_book_models,
    optimize_specific_book_model
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create optimizer to get available book names
    optimizer = HyperparameterOptimizer()
    available_books = optimizer.book_mapping['books']
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Neural Networks")
    parser.add_argument("--model_type", 
                       choices=["multi_label", "per_book", "both", "specific_book"], 
                       default="multi_label",
                       help="Model type to optimize")
    parser.add_argument("--book_name", 
                       type=str,
                       choices=available_books,
                       help="Specific book name for optimization (required for specific_book mode)")
    parser.add_argument("--n_trials", 
                       type=int, 
                       default=30,
                       help="Number of optimization trials")
    parser.add_argument("--timeout", 
                       type=int, 
                       default=1800,
                       help="Timeout in seconds per optimization")
    parser.add_argument("--quick", 
                       action="store_true",
                       help="Quick optimization with fewer trials")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        args.n_trials = 10
        args.timeout = 600
        logger.info("Running in quick mode with reduced trials and timeout")
    
    logger.info(f"Starting hyperparameter optimization for {args.model_type}")
    logger.info(f"Trials: {args.n_trials}, Timeout: {args.timeout}s")
    
    if args.model_type == "multi_label":
        logger.info("Optimizing multi-label neural network...")
        study = optimize_multi_label_model(args.n_trials, args.timeout)
        logger.info(f"Multi-label optimization completed!")
        logger.info(f"Best F1 Score: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
    elif args.model_type == "per_book":
        logger.info("Optimizing per-book models...")
        studies = optimize_per_book_models(args.n_trials, args.timeout)
        logger.info("Per-book optimization completed!")
        for book_col, study in studies.items():
            book_name = optimizer.book_names[book_col]
            logger.info(f"{book_name}: Best F1 Score: {study.best_trial.value:.4f}")
            
    elif args.model_type == "both":
        logger.info("Optimizing both model types...")
        
        # Multi-label optimization
        logger.info("1. Optimizing multi-label model...")
        multi_label_study = optimize_multi_label_model(args.n_trials, args.timeout)
        logger.info(f"Multi-label Best F1: {multi_label_study.best_trial.value:.4f}")
        
        # Per-book optimization
        logger.info("2. Optimizing per-book models...")
        per_book_studies = optimize_per_book_models(args.n_trials, args.timeout)
        
        logger.info("All optimizations completed!")
        logger.info("Summary:")
        logger.info(f"Multi-label: {multi_label_study.best_trial.value:.4f}")
        for book_col, study in per_book_studies.items():
            book_name = optimizer.book_names[book_col]
            logger.info(f"{book_name}: {study.best_trial.value:.4f}")
            
    elif args.model_type == "specific_book":
        if not args.book_name:
            logger.error("Error: --book_name is required for specific_book mode")
            logger.error(f"Available books: {', '.join(available_books)}")
            return
        
        logger.info(f"Optimizing specific book model: {args.book_name}")
        study = optimize_specific_book_model(args.book_name, args.n_trials, args.timeout)
        logger.info(f"{args.book_name} optimization completed!")
        logger.info(f"Best F1 Score: {study.best_trial.value:.4f}")
    
    logger.info("Optimization completed! Check optimization_results/ directory for detailed reports.")

if __name__ == "__main__":
    main() 