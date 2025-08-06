#!/usr/bin/env python3
"""
Setup script for Ollama local AI generation.
Installs Ollama and downloads a suitable model for sentence pair generation.
"""

import os
import sys
import subprocess
import requests
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama_installed():
    """Check if Ollama is already installed."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Ollama is already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    return False

def install_ollama():
    """Install Ollama based on the operating system."""
    system = os.uname().sysname.lower()
    
    logger.info(f"Installing Ollama on {system}...")
    
    if system == "darwin":  # macOS
        try:
            # Use Homebrew if available
            subprocess.run(['brew', 'install', 'ollama'], check=True)
            logger.info("Ollama installed via Homebrew")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Manual installation
            logger.info("Installing Ollama manually...")
            subprocess.run([
                'curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'
            ], shell=True, check=True)
    elif system == "linux":
        # Linux installation
        subprocess.run([
            'curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'
        ], shell=True, check=True)
    else:
        logger.error(f"Unsupported operating system: {system}")
        return False
    
    return True

def start_ollama_service():
    """Start the Ollama service."""
    try:
        # Check if service is already running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama service is already running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    logger.info("Starting Ollama service...")
    try:
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info("Ollama service started successfully")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        logger.error("Failed to start Ollama service")
        return False
        
    except Exception as e:
        logger.error(f"Error starting Ollama service: {e}")
        return False

def download_model(model_name="llama2:7b"):
    """Download a suitable model for sentence generation."""
    try:
        logger.info(f"Downloading model: {model_name}")
        
        # Check if model already exists
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            if any(model['name'] == model_name for model in models):
                logger.info(f"Model {model_name} already exists")
                return True
        
        # Download the model
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            logger.info(f"Model {model_name} downloaded successfully")
            return True
        else:
            logger.error(f"Failed to download model: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def test_ollama_generation():
    """Test if Ollama can generate sentence pairs."""
    try:
        test_prompt = """
Generate 2 pairs of semantically similar sentences about love.

Please respond with exactly 2 pairs in JSON format:
[
  {"sentence1": "first sentence", "sentence2": "similar sentence"},
  {"sentence1": "another sentence", "sentence2": "another similar sentence"}
]
"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:7b",
                "prompt": test_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Ollama generation test successful")
            logger.info(f"Response: {result['response'][:200]}...")
            return True
        else:
            logger.error(f"Generation test failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Ollama generation: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up Ollama for local AI generation...")
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        logger.info("Ollama not found, installing...")
        if not install_ollama():
            logger.error("Failed to install Ollama")
            sys.exit(1)
    
    # Step 2: Start Ollama service
    if not start_ollama_service():
        logger.error("Failed to start Ollama service")
        sys.exit(1)
    
    # Step 3: Download model
    if not download_model():
        logger.error("Failed to download model")
        sys.exit(1)
    
    # Step 4: Test generation
    if not test_ollama_generation():
        logger.error("Failed to test Ollama generation")
        sys.exit(1)
    
    logger.info("✅ Ollama setup completed successfully!")
    logger.info("You can now use local AI generation in your semantic model testing.")

if __name__ == "__main__":
    main() 