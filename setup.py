#!/usr/bin/env python3
"""
Setup script for the Book Sentence Classification project.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install project requirements."""
    logger.info("Installing project requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install requirements: {e}")
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data."""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt')
        logger.info("✓ NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to download NLTK data: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    logger.info("Creating project directories...")
    
    directories = [
        "data",
        "experiments/checkpoints",
        "experiments/results", 
        "experiments/visualizations",
        "experiments/logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {directory}")
    
    logger.info("✓ Directories created successfully")
    return True

def verify_setup():
    """Verify that the setup is complete."""
    logger.info("Verifying setup...")
    
    # Check if requirements are installed
    required_packages = [
        'torch', 'transformers', 'datasets', 'sklearn', 
        'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"✗ Missing packages: {missing_packages}")
        return False
    
    # Check if directories exist
    required_dirs = [
        "data", "experiments/checkpoints", "experiments/results",
        "experiments/visualizations", "experiments/logs"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        logger.error(f"✗ Missing directories: {missing_dirs}")
        return False
    
    logger.info("✓ Setup verification completed successfully")
    return True

def run_quick_test():
    """Run a quick test to ensure everything works."""
    logger.info("Running quick test...")
    
    try:
        # Test data preparation
        from data.prepare_data import InstitutionalBooksCorpusProcessor
        processor = InstitutionalBooksCorpusProcessor()
        dataset_dict, metadata = processor.create_dataset()
        logger.info("✓ Data preparation test passed")
        
        # Test model creation
        from models.constructive_model import ConstructiveLearningModel
        model = ConstructiveLearningModel()
        logger.info("✓ Model creation test passed")
        
        # Test evaluation utilities
        from utils.evaluation import calculate_metrics
        import numpy as np
        test_predictions = np.array([0, 1, 0, 1])
        test_labels = np.array([0, 1, 0, 1])
        test_probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        test_metadata = {'id_to_label': {'0': 'book_1', '1': 'book_2'}}
        
        metrics = calculate_metrics(test_predictions, test_labels, test_probabilities, test_metadata)
        logger.info("✓ Evaluation utilities test passed")
        
        logger.info("✓ All tests passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Quick test failed: {e}")
        return False

def main():
    """Main setup function."""
    
    logger.info("Setting up Book Sentence Classification Project")
    logger.info("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        logger.error("Setup failed at requirements installation")
        return False
    
    # Step 2: Download NLTK data
    if not download_nltk_data():
        logger.error("Setup failed at NLTK data download")
        return False
    
    # Step 3: Create directories
    if not create_directories():
        logger.error("Setup failed at directory creation")
        return False
    
    # Step 4: Verify setup
    if not verify_setup():
        logger.error("Setup verification failed")
        return False
    
    # Step 5: Run quick test
    if not run_quick_test():
        logger.error("Quick test failed")
        return False
    
    logger.info("=" * 50)
    logger.info("Setup completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run the demo: python demo.py")
    logger.info("2. Explore data: python data/prepare_data.py")
    logger.info("3. Train model: python models/train.py")
    logger.info("4. Evaluate model: python utils/evaluate.py")
    logger.info("5. Interactive demo: python demo.py --mode interactive")
    logger.info("")
    logger.info("Project structure:")
    logger.info("  data/ - Dataset and metadata")
    logger.info("  models/ - Model implementations")
    logger.info("  utils/ - Utility functions")
    logger.info("  experiments/ - Results and checkpoints")
    logger.info("  notebooks/ - Jupyter notebooks")
    logger.info("  configs/ - Configuration files")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 