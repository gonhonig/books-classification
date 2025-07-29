#!/usr/bin/env python3
"""
Fix CUDA version mismatch in Google Colab.
This script reinstalls PyTorch with compatible versions.
"""

import subprocess
import sys

def fix_cuda_versions():
    """Fix CUDA version mismatches in Colab."""
    print("🔧 Fixing CUDA version issues...")
    
    # Uninstall existing PyTorch packages
    print("📦 Uninstalling existing PyTorch packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                   capture_output=True)
    
    # Install PyTorch with compatible CUDA version
    print("📦 Installing PyTorch with CUDA 11.8...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    # Install compatible transformers version
    print("📦 Installing compatible transformers...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "transformers==4.30.2", "datasets==2.12.0", "tokenizers==0.13.3"
    ], check=True)
    
    # Install other dependencies
    print("📦 Installing other dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "accelerate==0.20.3", "deepspeed==0.9.5"
    ], check=True)
    
    print("✅ CUDA version fix completed!")

def verify_installation():
    """Verify the installation worked."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed successfully")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__} installed successfully")
        
        return True
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False

if __name__ == "__main__":
    fix_cuda_versions()
    if verify_installation():
        print("🎉 CUDA version fix successful! You can now run training.")
    else:
        print("⚠️  CUDA version fix may have issues. Please restart the runtime.") 