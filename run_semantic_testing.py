#!/usr/bin/env python3
"""
Script to set up Ollama and run semantic model testing with book-based prompts.
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_setup():
    """Run the Ollama setup."""
    logger.info("Setting up Ollama for local AI generation...")
    
    try:
        # Run the setup script
        result = subprocess.run([sys.executable, "setup_ollama.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Ollama setup completed successfully!")
            return True
        else:
            logger.error(f"❌ Ollama setup failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running setup: {e}")
        return False

def run_semantic_testing():
    """Run the semantic model testing."""
    logger.info("Running semantic model testing with book-based prompts...")
    
    try:
        # Run the semantic testing
        result = subprocess.run([sys.executable, "test_semantic_models.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Semantic model testing completed successfully!")
            print(result.stdout)
            return True
        else:
            logger.error(f"❌ Semantic model testing failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running semantic testing: {e}")
        return False

def main():
    """Main function to run setup and testing."""
    print("🚀 Setting up Ollama and running semantic model testing...")
    print("=" * 60)
    
    # Step 1: Setup Ollama
    if not run_setup():
        print("❌ Setup failed. Please check the logs above.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Step 2: Run semantic testing
    if not run_semantic_testing():
        print("❌ Semantic testing failed. Please check the logs above.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All tasks completed successfully!")
    print("📁 Check the 'experiments/model_selection' directory for results.")

if __name__ == "__main__":
    main() 