#!/usr/bin/env python3
"""
Script to prepare project files for Google Colab.
Creates a zip file with all necessary files for cloud training.
"""

import os
import zipfile
from pathlib import Path

def create_colab_package():
    """Create a zip file with all necessary project files for Colab."""
    
    # Files to include in the package
    files_to_include = [
        # Core project files
        'configs/config.yaml',
        'data/prepare_data.py',
        'models/constructive_model.py',
        'models/train.py',
        'train_cloud.py',
        'requirements-cloud.txt',
        'README_CLOUD.md',
        'test_prediction.py',
        
        # Utils
        'utils/data_utils.py',
        'utils/evaluate.py',
        'utils/evaluation.py',
        'utils/visualization.py',
        
        # Demo
        'demo.py',
        'setup.py',
        'README.md'
    ]
    
    # Directories to include
    dirs_to_include = [
        'configs',
        'data',
        'models', 
        'utils'
    ]
    
    # Create zip file
    zip_filename = 'books_classification_colab.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add individual files
        for file_path in files_to_include:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                print(f"Added: {file_path}")
            else:
                print(f"Warning: {file_path} not found")
        
        # Add directories
        for dir_path in dirs_to_include:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Skip __pycache__ and other unnecessary files
                        if '__pycache__' not in file_path and '.pyc' not in file:
                            zipf.write(file_path, file_path)
                            print(f"Added: {file_path}")
    
    print(f"\n✅ Colab package created: {zip_filename}")
    print(f"📦 File size: {os.path.getsize(zip_filename) / 1024 / 1024:.1f} MB")
    print("\n📤 Ready to upload to Google Colab!")
    
    return zip_filename

if __name__ == "__main__":
    create_colab_package() 