#!/usr/bin/env python3
"""
Utility functions to find the latest optimization studies and extract study information.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def find_latest_optimization_studies() -> Dict[str, str]:
    """
    Find the latest optimization studies in the optimization_results directory.
    
    Returns:
        Dictionary with keys 'multi_label' and/or 'per_book' containing study names
    """
    results_dir = Path("optimization_results")
    if not results_dir.exists():
        logger.warning("optimization_results directory not found")
        return {}
    
    latest_studies = {}
    
    # Find multi-label studies
    multi_label_studies = []
    for file in results_dir.glob("multi_label_optimization_*.pkl"):
        study_name = file.stem
        multi_label_studies.append((study_name, file.stat().st_mtime))
    
    if multi_label_studies:
        # Sort by modification time (newest first)
        multi_label_studies.sort(key=lambda x: x[1], reverse=True)
        latest_studies['multi_label'] = multi_label_studies[0][0]
        logger.info(f"Found latest multi-label study: {latest_studies['multi_label']}")
    
    # Find per-book studies
    per_book_studies = []
    for file in results_dir.glob("per_book_optimization_*.pkl"):
        study_name = file.stem
        per_book_studies.append((study_name, file.stat().st_mtime))
    
    if per_book_studies:
        # Sort by modification time (newest first)
        per_book_studies.sort(key=lambda x: x[1], reverse=True)
        latest_studies['per_book'] = per_book_studies[0][0]
        logger.info(f"Found latest per-book study: {latest_studies['per_book']}")
    
    return latest_studies

def get_study_info(study_name: str) -> Dict[str, Any]:
    """
    Get information about a specific study.
    
    Args:
        study_name: Name of the study
        
    Returns:
        Dictionary containing study information
    """
    results_dir = Path("optimization_results")
    study_file = results_dir / f"{study_name}.pkl"
    
    if not study_file.exists():
        return {"error": f"Study file not found: {study_file}"}
    
    try:
        with open(study_file, 'rb') as f:
            study = pickle.load(f)
        
        # Extract study information
        info = {
            "study_name": study_name,
            "best_f1": study.best_trial.value if study.best_trial else None,
            "parameters": study.best_trial.params if study.best_trial else {},
            "n_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state.name == 'COMPLETE']),
            "failed_trials": len([t for t in study.trials if t.state.name == 'FAILED']),
            "study_file": str(study_file)
        }
        
        return info
        
    except Exception as e:
        return {"error": f"Failed to load study: {e}"}

def get_book_study_info(base_study_name: str, book_col: str) -> Dict[str, Any]:
    """
    Get information about a specific book study.
    
    Args:
        base_study_name: Base name of the study
        book_col: Book column name (e.g., 'book_1')
        
    Returns:
        Dictionary containing study information
    """
    study_name = f"{base_study_name}_{book_col}"
    return get_study_info(study_name)

def list_all_studies() -> Dict[str, list]:
    """
    List all available studies in the optimization_results directory.
    
    Returns:
        Dictionary with 'multi_label' and 'per_book' lists of study names
    """
    results_dir = Path("optimization_results")
    if not results_dir.exists():
        return {"multi_label": [], "per_book": []}
    
    studies = {"multi_label": [], "per_book": []}
    
    # Find multi-label studies
    for file in results_dir.glob("multi_label_optimization_*.pkl"):
        studies["multi_label"].append(file.stem)
    
    # Find per-book studies
    for file in results_dir.glob("per_book_optimization_*.pkl"):
        studies["per_book"].append(file.stem)
    
    return studies

if __name__ == "__main__":
    # Test the functions
    print("Finding latest studies...")
    latest = find_latest_optimization_studies()
    print(f"Latest studies: {latest}")
    
    print("\nListing all studies...")
    all_studies = list_all_studies()
    print(f"All studies: {all_studies}")
    
    if latest.get('multi_label'):
        print(f"\nMulti-label study info:")
        info = get_study_info(latest['multi_label'])
        print(json.dumps(info, indent=2)) 