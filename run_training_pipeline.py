"""
Main Training Pipeline Runner
Executes the complete multi-label classification training pipeline.
"""

import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name: str, description: str):
    """Run a Python script and handle errors."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✅ {description} completed successfully!")
        if result.stdout:
            logger.info("Output:")
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed!")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    """Run the complete training pipeline."""
    logger.info("🚀 Starting Multi-Label Classification Training Pipeline")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if dataset exists
    dataset_path = Path("data/semantic_augmented/semantic_augmented_dataset.csv")
    if not dataset_path.exists():
        logger.error(f"❌ Dataset not found: {dataset_path}")
        logger.error("Please ensure the semantic augmented dataset is available.")
        return False
    
    logger.info(f"✅ Dataset found: {dataset_path}")
    
    # Step 1: Train models with default parameters
    if not run_script("train_multi_label_models.py", "Training models with default parameters"):
        return False
    
    # Step 2: Hyperparameter optimization
    if not run_script("optimize_hyperparameters.py", "Hyperparameter optimization"):
        return False
    
    # Step 3: Evaluate optimized models
    if not run_script("evaluate_optimized_models.py", "Evaluating optimized models"):
        return False
    
    logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("\nResults are available in:")
    logger.info("  - trained_models/ (default parameter models)")
    logger.info("  - optimization_results/ (optimized models)")
    logger.info("  - evaluation_results/ (comparison results)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 