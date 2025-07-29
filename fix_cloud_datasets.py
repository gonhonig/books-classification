#!/usr/bin/env python3
"""
Fix datasets version issues for cloud training.
This script installs compatible versions and handles the protocol issue.
"""

import subprocess
import sys

def fix_datasets_issues():
    """Fix datasets version and protocol issues."""
    print("🔧 Fixing datasets version issues...")
    
    # Uninstall problematic packages
    print("📦 Uninstalling problematic packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "datasets", "transformers"], 
                   capture_output=True)
    
    # Install compatible versions
    print("📦 Installing compatible versions...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "datasets==2.14.0", "transformers==4.30.2", "tokenizers==0.13.3"
    ], check=True)
    
    # Install other dependencies
    print("📦 Installing other dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "accelerate==0.20.3", "deepspeed==0.9.5",
        "wandb", "tqdm", "PyYAML", "nltk", "scikit-learn", "pandas", "numpy"
    ], check=True)
    
    print("✅ Datasets version fix completed!")

def verify_installation():
    """Verify the installation worked."""
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__} installed successfully")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__} installed successfully")
        
        import torch
        print(f"✅ PyTorch {torch.__version__} installed successfully")
        
        # Test dataset loading
        from datasets import load_from_disk
        print("✅ Dataset loading functions available")
        
        return True
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False

if __name__ == "__main__":
    fix_datasets_issues()
    if verify_installation():
        print("🎉 Datasets version fix successful! You can now run cloud training.")
    else:
        print("⚠️  Datasets version fix may have issues. Please restart the runtime.") 