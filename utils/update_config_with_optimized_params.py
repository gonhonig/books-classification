#!/usr/bin/env python3
"""
Script to automatically extract the best parameters from the latest optimization study
and update the configuration file for training.
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.find_latest_optimization import find_latest_optimization_studies, get_study_info

logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/optimized_params_config.yaml") -> Dict[str, Any]:
    """Load the current configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def extract_hidden_dims_from_params(params: Dict[str, Any]) -> list:
    """Extract hidden_dims list from optimization parameters."""
    n_layers = params.get('n_layers', 2)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(params[f'hidden_dim_{i}'])
    return hidden_dims

def update_multi_label_config(config: Dict[str, Any], study_name: str) -> Dict[str, Any]:
    """Update multi-label configuration with optimized parameters."""
    logger.info(f"Updating multi-label config with study: {study_name}")
    
    # Get study info
    study_info = get_study_info(study_name)
    if "error" in study_info:
        logger.error(f"Failed to get study info: {study_info['error']}")
        return config
    
    params = study_info['parameters']
    
    # Update multi-label config
    config['multi_label']['architecture']['hidden_dims'] = extract_hidden_dims_from_params(params)
    config['multi_label']['architecture']['dropout_rate'] = params['dropout_rate']
    config['multi_label']['training']['epochs'] = params['epochs']
    config['multi_label']['training']['batch_size'] = params['batch_size']
    config['multi_label']['training']['learning_rate'] = params['learning_rate']
    config['multi_label']['training']['patience'] = params['patience']
    config['multi_label']['training']['weight_decay'] = params['weight_decay']
    
    logger.info(f"Updated multi-label config with F1 score: {study_info.get('best_f1', 'N/A')}")
    return config

def update_per_book_config(config: Dict[str, Any], base_study_name: str) -> Dict[str, Any]:
    """Update per-book configurations with optimized parameters."""
    logger.info(f"Updating per-book config with base study: {base_study_name}")
    
    # Load book mapping to get the correct book columns
    mapping_file = Path("data/semantic_augmented/book_to_label_mapping.json")
    if not mapping_file.exists():
        logger.error(f"Book mapping file not found: {mapping_file}")
        return config
    
    try:
        with open(mapping_file, 'r') as f:
            book_mapping = json.load(f)
        
        # Generate book columns from the mapping
        book_columns = []
        for book_name in book_mapping['books']:
            # Convert book name to column format (replace spaces with underscores, remove apostrophes)
            clean_name = book_name.replace(' ', '_').replace("'", '')
            book_col = f"book_{clean_name}"
            book_columns.append(book_col)
        
        logger.info(f"Using book columns: {book_columns}")
        
    except Exception as e:
        logger.error(f"Failed to load book mapping: {e}")
        return config
    
    # Get all available studies to find the ones for each book
    from utils.find_latest_optimization import list_all_studies
    all_studies = list_all_studies()
    per_book_studies = all_studies.get('per_book', [])
    
    for book_col in book_columns:
        book_name = book_col.replace('book_', '').replace('_', ' ')
        logger.info(f"Processing {book_col} ({book_name})")
        
        # Find the study for this book
        book_study = None
        for study_name in per_book_studies:
            # Extract the book name from the study name
            if 'book_' in study_name:
                study_book_name = study_name.split('book_')[-1].replace('_', ' ')
                if study_book_name == book_name:
                    book_study = study_name
                    break
        
        if book_study is None:
            logger.warning(f"No study found for {book_col} ({book_name})")
            continue
        
        try:
            # Get study info
            study_info = get_study_info(book_study)
            if "error" in study_info:
                logger.warning(f"Failed to get study info for {book_col}: {study_info['error']}")
                continue
            
            params = study_info['parameters']
            
            # Update book config
            config['per_book'][book_col]['architecture']['hidden_dims'] = extract_hidden_dims_from_params(params)
            config['per_book'][book_col]['architecture']['dropout_rate'] = params['dropout_rate']
            config['per_book'][book_col]['training']['epochs'] = params['epochs']
            config['per_book'][book_col]['training']['batch_size'] = params['batch_size']
            config['per_book'][book_col]['training']['learning_rate'] = params['learning_rate']
            config['per_book'][book_col]['training']['patience'] = params['patience']
            config['per_book'][book_col]['training']['weight_decay'] = params['weight_decay']
            
            logger.info(f"Updated {book_col} config with F1 score: {study_info.get('best_f1', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to update config for {book_col}: {e}")
    
    return config

def update_metadata(config: Dict[str, Any], latest_studies: Dict[str, str]) -> Dict[str, Any]:
    """Update metadata with latest study information."""
    config['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['metadata']['optimization_date'] = datetime.now().strftime("%Y-%m-%d")
    
    if 'multi_label' in latest_studies:
        config['metadata']['multi_label_study'] = latest_studies['multi_label']
    
    if 'per_book' in latest_studies:
        config['metadata']['per_book_study'] = latest_studies['per_book']
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "configs/optimized_params_config.yaml"):
    """Save the updated configuration to file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="Update config with latest optimized parameters")
    parser.add_argument("--config", type=str, default="configs/optimized_params_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--force", action="store_true", 
                       help="Force update even if no new studies found")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without saving")
    
    args = parser.parse_args()
    
    try:
        # Load current config
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
        
        # Find latest studies
        latest_studies = find_latest_optimization_studies()
        logger.info(f"Found latest studies: {latest_studies}")
        
        if not latest_studies and not args.force:
            logger.warning("No optimization studies found. Use --force to update with default values.")
            return
        
        # Update multi-label config if available
        if 'multi_label' in latest_studies:
            config = update_multi_label_config(config, latest_studies['multi_label'])
        else:
            logger.warning("No multi-label optimization study found")
        
        # Update per-book config if available
        if 'per_book' in latest_studies:
            config = update_per_book_config(config, latest_studies['per_book'])
        else:
            logger.warning("No per-book optimization study found")
        
        # Update metadata
        config = update_metadata(config, latest_studies)
        
        if args.dry_run:
            print("DRY RUN - Configuration that would be saved:")
            print("=" * 50)
            print(yaml.dump(config, default_flow_style=False, indent=2))
        else:
            # Save updated config
            save_config(config, args.config)
            logger.info("Configuration updated successfully!")
            
            # Print summary
            print("\n📋 Configuration Update Summary:")
            print("=" * 40)
            if 'multi_label' in latest_studies:
                print(f"✅ Multi-label: {latest_studies['multi_label']}")
            if 'per_book' in latest_studies:
                print(f"✅ Per-book: {latest_studies['per_book']}")
            print(f"📅 Updated: {config['metadata']['last_updated']}")
    
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise

if __name__ == "__main__":
    main() 