#!/usr/bin/env python3
"""
Step 2: Semantic Embedding Model Selection
Test 4 candidate models and select the best one based on similarity performance.
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticModelTester:
    """Test semantic embedding models for similarity performance."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the model tester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.results_dir = Path("experiments/model_selection")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Setup device
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_data(self):
        """Load the processed dataset."""
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load processed dataset
        from datasets import load_from_disk
        self.dataset = load_from_disk(str(self.data_dir / "dataset"))
        
        logger.info(f"Loaded dataset with {len(self.dataset['train'])} training samples")
        logger.info(f"Books: {self.metadata['books']}")
    
    def load_similarity_test_pairs(self) -> List[Dict]:
        """Load test pairs for similarity evaluation."""
        logger.info("Loading similarity test pairs...")
        
        test_pairs_path = self.data_dir / "similarity_test_pairs.json"
        if not test_pairs_path.exists():
            raise FileNotFoundError(
                f"Test pairs file not found: {test_pairs_path}\n"
                "Please run 'python create_similarity_pairs.py' first to generate test pairs."
            )
        
        with open(test_pairs_path, 'r') as f:
            test_pairs = json.load(f)
        
        logger.info(f"Loaded {len(test_pairs)} test pairs")
        return test_pairs
    
    
    def test_model(self, model_name: str) -> Dict:
        """Test a single semantic embedding model."""
        logger.info(f"Testing model: {model_name}")
        
        try:
            # Load model
            start_time = time.time()
            model = SentenceTransformer(model_name, device=self.device)
            load_time = time.time() - start_time
            
            # Get test pairs (loaded from file)
            test_pairs = self.load_similarity_test_pairs()
            
            # Compute embeddings and similarities
            similarities = []
            embedding_times = []
            
            for pair in tqdm(test_pairs, desc=f"Testing {model_name}"):
                # Compute embeddings
                start_time = time.time()
                embeddings = model.encode([pair['sentence1'], pair['sentence2']])
                embedding_time = time.time() - start_time
                embedding_times.append(embedding_time)
                
                # Compute cosine similarity
                similarity = cosine_similarity(
                    embeddings[0].reshape(1, -1), 
                    embeddings[1].reshape(1, -1)
                )[0][0]
                
                similarities.append(similarity)
            
            # Calculate metrics
            similar_scores = [s for i, s in enumerate(similarities) if test_pairs[i]['type'] == 'similar']
            dissimilar_scores = [s for i, s in enumerate(similarities) if test_pairs[i]['type'] == 'dissimilar']
            
            # Calculate accuracy (how well model distinguishes similar vs dissimilar)
            threshold = 0.5
            correct_predictions = 0
            total_predictions = 0
            
            for score in similar_scores:
                if score > threshold:
                    correct_predictions += 1
                total_predictions += 1
            
            for score in dissimilar_scores:
                if score <= threshold:
                    correct_predictions += 1
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            # Calculate other metrics
            avg_similar_score = np.mean(similar_scores)
            avg_dissimilar_score = np.mean(dissimilar_scores)
            score_separation = avg_similar_score - avg_dissimilar_score
            
            # Model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
            # Average embedding time
            avg_embedding_time = np.mean(embedding_times)
            
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'avg_similar_score': avg_similar_score,
                'avg_dissimilar_score': avg_dissimilar_score,
                'score_separation': score_separation,
                'load_time': load_time,
                'avg_embedding_time': avg_embedding_time,
                'model_size_mb': model_size_mb,
                'total_pairs': len(test_pairs),
                'similar_pairs': len(similar_scores),
                'dissimilar_pairs': len(dissimilar_scores)
            }
            
            logger.info(f"Model {model_name} results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Avg similar score: {avg_similar_score:.4f}")
            logger.info(f"  Avg dissimilar score: {avg_dissimilar_score:.4f}")
            logger.info(f"  Score separation: {score_separation:.4f}")
            logger.info(f"  Model size: {model_size_mb:.2f} MB")
            logger.info(f"  Avg embedding time: {avg_embedding_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'accuracy': 0.0,
                'avg_similar_score': 0.0,
                'avg_dissimilar_score': 0.0,
                'score_separation': 0.0,
                'load_time': 0.0,
                'avg_embedding_time': 0.0,
                'model_size_mb': 0.0
            }
    
    def test_all_models(self) -> Dict:
        """Test all candidate models."""
        logger.info("Testing all semantic embedding models...")
        
        model_names = self.config['semantic_models']['candidates']
        results = {}
        
        for model_name in model_names:
            result = self.test_model(model_name)
            results[model_name] = result
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['accuracy'])
            logger.info(f"Best model: {best_model} (accuracy: {valid_results[best_model]['accuracy']:.4f})")
        else:
            best_model = None
            logger.error("No models completed successfully")
        
        # Save results
        results_summary = {
            'results': results,
            'best_model': best_model,
            'test_timestamp': time.time()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_summary_serializable = convert_numpy_types(results_summary)
        
        results_path = self.results_dir / "semantic_model_comparison.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary_serializable, f, indent=2)
        
        # Create comparison table
        self._create_comparison_table(results)
        
        return results_summary
    
    def _create_comparison_table(self, results: Dict):
        """Create a comparison table of all models."""
        # Filter out failed models
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid results to create comparison table")
            return
        
        # Create DataFrame
        df_data = []
        for model_name, result in valid_results.items():
            df_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Avg Similar Score': f"{result['avg_similar_score']:.4f}",
                'Avg Dissimilar Score': f"{result['avg_dissimilar_score']:.4f}",
                'Score Separation': f"{result['score_separation']:.4f}",
                'Model Size (MB)': f"{result['model_size_mb']:.2f}",
                'Avg Embedding Time (s)': f"{result['avg_embedding_time']:.4f}"
            })
        
        df = pd.DataFrame(df_data)
        
        # Save table
        table_path = self.results_dir / "model_comparison_table.csv"
        df.to_csv(table_path, index=False)
        
        # Print table
        print("\n" + "="*80)
        print("SEMANTIC MODEL COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        logger.info(f"Comparison table saved to {table_path}")

def test_semantic_models(config_path: str = "configs/config.yaml"):
    """Test all semantic embedding models."""
    tester = SemanticModelTester(config_path=config_path)
    results = tester.test_all_models()
    
    print(f"\n✅ Semantic model testing completed!")
    print(f"📁 Results saved to: {tester.results_dir}")
    
    if results['best_model']:
        print(f"🏆 Best model: {results['best_model']}")
        best_result = results['results'][results['best_model']]
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Score separation: {best_result['score_separation']:.4f}")
    
    return results

def main():
    """Main function to test semantic models."""
    parser = argparse.ArgumentParser(description="Test semantic embedding models")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    results = test_semantic_models(config_path=args.config)
    
    if not results:
        sys.exit(1)

if __name__ == "__main__":
    main() 