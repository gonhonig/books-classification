#!/usr/bin/env python3
"""
Complete Books Classification Pipeline
====================================

This script runs the entire pipeline from data preparation to final report generation.
It includes:
1. Data preparation and preprocessing
2. Model training and fine-tuning
3. Model evaluation and testing
4. Comprehensive report generation
5. Visualization creation

Usage:
    python run_full_pipeline.py [--force] [--skip-training] [--skip-evaluation]
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Runs the complete books classification pipeline."""
    
    def __init__(self, force=False, skip_training=False, skip_evaluation=False):
        self.force = force
        self.skip_training = skip_training
        self.skip_evaluation = skip_evaluation
        self.start_time = time.time()
        
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("🚀 Starting Books Classification Pipeline")
        logger.info("=" * 60)
        logger.info(f"📚 Selected Books: {self.config['data']['selected_books']}")
        logger.info(f"🔧 Force mode: {self.force}")
        logger.info(f"⏭️ Skip training: {self.skip_training}")
        logger.info(f"⏭️ Skip evaluation: {self.skip_evaluation}")
        
        # Log model configuration
        if 'model' in self.config:
            logger.info(f"🤖 Model: {self.config['model']['encoder']['model_name']}")
            logger.info(f"📏 Embedding dimension: {self.config['model']['encoder']['hidden_size']}")
        
        # Log training configuration
        if 'training' in self.config:
            logger.info(f"📦 Batch size: {self.config['training']['batch_size']}")
            logger.info(f"📈 Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
            logger.info(f"🔄 Max grad norm: {self.config['training']['max_grad_norm']}")
        
        # Log training phases
        if 'model' in self.config and 'training_phases' in self.config['model']:
            logger.info(f"🎯 Training phases: {len(self.config['model']['training_phases'])}")
            for phase in self.config['model']['training_phases']:
                logger.info(f"  - {phase['name']}: {phase['epochs']} epochs, lr={phase['learning_rate']}")
        
        logger.info("=" * 60)
    
    def run_command(self, command, description):
        """Run a command and log the result."""
        start_time = time.time()
        logger.info(f"🔄 {description}")
        logger.info(f"Command: {command}")
        logger.info(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ {description} completed successfully")
            logger.info(f"⏱️ Duration: {duration:.2f} seconds")
            
            # Log command output
            if result.stdout:
                logger.info(f"📝 {description} output:")
                # Split output into lines and log each line
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.strip():  # Only log non-empty lines
                        logger.info(f"  {line}")
            
            return True
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"❌ {description} failed")
            logger.error(f"⏱️ Duration: {duration:.2f} seconds")
            logger.error(f"Error code: {e.returncode}")
            
            # Log stderr output
            if e.stderr:
                logger.error(f"📝 {description} error output:")
                error_lines = e.stderr.strip().split('\n')
                for line in error_lines:
                    if line.strip():  # Only log non-empty lines
                        logger.error(f"  {line}")
            
            # Log stdout if available (might contain useful info even on error)
            if e.stdout:
                logger.info(f"📝 {description} partial output:")
                output_lines = e.stdout.strip().split('\n')
                for line in output_lines:
                    if line.strip():  # Only log non-empty lines
                        logger.info(f"  {line}")
            
            return False
    
    def check_prerequisites(self):
        """Check if all required files and dependencies are available."""
        logger.info("🔍 Checking prerequisites...")
        
        # Log environment information
        import os
        logger.info(f"📁 Working directory: {os.getcwd()}")
        logger.info(f"🐍 Python version: {sys.version}")
        logger.info(f"🔧 Platform: {sys.platform}")
        
        required_files = [
            "configs/config.yaml",
            "data/prepare_data.py",
            "fine_tune_models.py",
            "evaluate_models.py",
            "create_final_report.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
            else:
                logger.info(f"✅ Found: {file_path}")
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def prepare_data(self):
        """Step 1: Prepare and preprocess the data."""
        logger.info("📊 Step 1: Data Preparation")
        
        force_flag = "--force" if self.force else ""
        command = f"python data/prepare_data.py {force_flag}"
        
        if not self.run_command(command, "Data preparation"):
            logger.error("❌ Data preparation failed. Stopping pipeline.")
            return False
        
        # Verify data was created
        required_data_files = [
            "data/metadata.json",
            "data/processed_dataset/dataset_dict.json"
        ]
        
        for file_path in required_data_files:
            if not Path(file_path).exists():
                logger.error(f"❌ Required data file not found: {file_path}")
                return False
        
        logger.info("✅ Data preparation completed successfully")
        return True
    
    def train_models(self):
        """Step 2: Train and fine-tune the models."""
        if self.skip_training:
            logger.info("⏭️ Skipping model training as requested")
            return True
        
        logger.info("🤖 Step 2: Model Training and Fine-tuning")
        
        command = "python fine_tune_models.py"
        
        if not self.run_command(command, "Model training and fine-tuning"):
            logger.error("❌ Model training failed. Stopping pipeline.")
            return False
        
        # Verify model was created
        model_path = Path("experiments/fine_tuned_model.pt")
        if not model_path.exists():
            logger.error("❌ Trained model not found")
            return False
        
        logger.info("✅ Model training completed successfully")
        return True
    
    def evaluate_models(self):
        """Step 3: Evaluate the trained models."""
        if self.skip_evaluation:
            logger.info("⏭️ Skipping model evaluation as requested")
            return True
        
        logger.info("📈 Step 3: Model Evaluation")
        
        command = "python evaluate_models.py"
        
        if not self.run_command(command, "Model evaluation"):
            logger.error("❌ Model evaluation failed. Stopping pipeline.")
            return False
        
        # Verify evaluation results were created
        required_eval_files = [
            "experiments/evaluation_results/evaluation_report.txt",
            "experiments/evaluation_results/evaluation_report.json"
        ]
        
        for file_path in required_eval_files:
            if not Path(file_path).exists():
                logger.warning(f"⚠️ Evaluation file not found: {file_path}")
        
        logger.info("✅ Model evaluation completed successfully")
        return True
    
    def generate_report(self):
        """Step 4: Generate comprehensive final report."""
        logger.info("📋 Step 4: Generating Comprehensive Report")
        
        command = "python create_final_report.py"
        
        if not self.run_command(command, "Report generation"):
            logger.error("❌ Report generation failed.")
            return False
        
        # Verify report was created
        report_path = Path("experiments/comprehensive_report.txt")
        if not report_path.exists():
            logger.error("❌ Comprehensive report not found")
            return False
        
        logger.info("✅ Report generation completed successfully")
        return True
    
    def create_summary(self):
        """Create a summary of the pipeline execution."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        logger.info("📊 Pipeline Summary")
        logger.info("=" * 50)
        logger.info(f"Total execution time: {duration:.2f} seconds")
        logger.info(f"Configuration: {self.config['data']['selected_books']}")
        
        # Check final outputs
        outputs = {
            "Data": Path("data/metadata.json").exists(),
            "Trained Model": Path("experiments/fine_tuned_model.pt").exists(),
            "Evaluation Results": Path("experiments/evaluation_results").exists(),
            "Comprehensive Report": Path("experiments/comprehensive_report.txt").exists(),
            "Visualizations": Path("experiments/results_summary.png").exists()
        }
        
        # Adjust expectations based on skipped steps
        if self.skip_training:
            outputs["Trained Model"] = True  # Don't expect this if skipped
        if self.skip_evaluation:
            outputs["Evaluation Results"] = True  # Don't expect this if skipped
        
        logger.info("Generated outputs:")
        for output, exists in outputs.items():
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {output}")
        
        # Show final report location
        if Path("experiments/comprehensive_report.txt").exists():
            logger.info("\n📄 Final report available at: experiments/comprehensive_report.txt")
        
        return all(outputs.values())
    
    def run(self):
        """Run the complete pipeline."""
        logger.info("🚀 Starting complete pipeline execution")
        logger.info("=" * 60)
        
        # Step 0: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 1: Prepare data
        if not self.prepare_data():
            return False
        
        # Step 2: Train models
        if not self.train_models():
            return False
        
        # Step 3: Evaluate models
        if not self.evaluate_models():
            return False
        
        # Step 4: Generate report
        if not self.generate_report():
            return False
        
        # Create summary
        success = self.create_summary()
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        return success

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete books classification pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of data files"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation step"
    )
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = PipelineRunner(
        force=args.force,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
    
    # Run pipeline
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 