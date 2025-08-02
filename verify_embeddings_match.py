"""
Verify that embeddings in cache match the sentences in the dataset.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import random

def verify_embeddings_match():
    """Verify that cached embeddings match the dataset sentences."""
    
    print("Loading dataset...")
    df = pd.read_csv('data/semantic_augmented/semantic_augmented_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    
    print("Loading cached embeddings...")
    cached_embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')['embeddings']
    print(f"Cached embeddings shape: {cached_embeddings.shape}")
    
    # Check shapes match
    if len(df) != len(cached_embeddings):
        print(f"ERROR: Shape mismatch! Dataset: {len(df)}, Embeddings: {len(cached_embeddings)}")
        return False
    
    print("✓ Shapes match!")
    
    # Randomly sample 5 sentences for verification
    random.seed(42)
    sample_indices = random.sample(range(len(df)), 5)
    
    print(f"\nVerifying embeddings for {len(sample_indices)} random sentences...")
    
    # Load the model to compute fresh embeddings
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    for i, idx in enumerate(sample_indices):
        sentence = df.iloc[idx]['sentence']
        cached_embedding = cached_embeddings[idx]
        
        print(f"\n{i+1}. Sentence: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
        print(f"   Index: {idx}")
        
        # Compute fresh embedding
        fresh_embedding = model.encode([sentence])[0]
        
        # Compare embeddings
        similarity = np.dot(cached_embedding, fresh_embedding) / (np.linalg.norm(cached_embedding) * np.linalg.norm(fresh_embedding))
        
        print(f"   Cached embedding norm: {np.linalg.norm(cached_embedding):.6f}")
        print(f"   Fresh embedding norm: {np.linalg.norm(fresh_embedding):.6f}")
        print(f"   Cosine similarity: {similarity:.6f}")
        
        if similarity > 0.9999:
            print("   ✓ Embeddings match!")
        else:
            print("   ✗ Embeddings don't match!")
            return False
    
    print("\n✓ All sampled embeddings match!")
    return True

if __name__ == "__main__":
    success = verify_embeddings_match()
    if success:
        print("\n🎉 Verification successful! The cached embeddings match the dataset.")
    else:
        print("\n❌ Verification failed! There's a mismatch between embeddings and dataset.") 