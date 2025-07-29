#!/usr/bin/env python3
"""
Quick start script for Google Colab.
Automatically sets up and runs training for the books classification project.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  No GPU detected! Please enable GPU in Runtime > Change runtime type")
        return False

def install_dependencies():
    """Install required packages."""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers datasets tokenizers",
        "nltk scikit-learn pandas numpy matplotlib seaborn",
        "wandb tensorboard tqdm PyYAML",
        "accelerate deepspeed"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run(["pip", "install"] + package.split(), check=True)
    
    print("✅ Dependencies installed!")

def download_nltk_data():
    """Download NLTK data."""
    print("📥 Downloading NLTK data...")
    
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    print("✅ NLTK data downloaded!")

def prepare_data():
    """Prepare the dataset."""
    print("🗂️  Preparing data...")
    
    if not os.path.exists("data/prepare_data.py"):
        print("❌ data/prepare_data.py not found!")
        return False
    
    subprocess.run(["python", "data/prepare_data.py"], check=True)
    print("✅ Data preparation completed!")
    return True

def start_training(epochs=5):
    """Start training."""
    print(f"🚀 Starting training for {epochs} epochs...")
    
    if not os.path.exists("train_cloud.py"):
        print("❌ train_cloud.py not found!")
        return False
    
    subprocess.run(["python", "train_cloud.py", "--epochs", str(epochs)], check=True)
    print("✅ Training completed!")
    return True

def test_model():
    """Test the trained model."""
    print("🧪 Testing model...")
    
    if not os.path.exists("test_prediction.py"):
        print("❌ test_prediction.py not found!")
        return False
    
    subprocess.run(["python", "test_prediction.py"], check=True)
    print("✅ Model testing completed!")
    return True

def main():
    """Main function."""
    print("🚀 Books Classification - Colab Quick Start")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Download NLTK data
    download_nltk_data()
    
    # Prepare data
    if not prepare_data():
        print("❌ Data preparation failed!")
        return
    
    # Start training
    if not start_training(epochs=5):
        print("❌ Training failed!")
        return
    
    # Test model
    if not test_model():
        print("❌ Model testing failed!")
        return
    
    print("\n🎉 All done! Training completed successfully!")
    print("📊 Check experiments/ directory for results")
    print("📁 Download results using the download cell in the notebook")

if __name__ == "__main__":
    main() 