#!/usr/bin/env python3
"""
Simple test script for book sentence classification.
"""

import torch
from transformers import AutoTokenizer
from models.constructive_model import ConstructiveLearningModel
import json

def test_prediction():
    """Test book sentence classification with a sample sentence."""
    
    print("Testing Book Sentence Classification")
    print("=" * 50)
    
    # Load metadata to get class names
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)
    
    class_names = metadata['id_to_label']
    print(f"Available books: {list(class_names.values())}")
    print()
    
    # Sample sentences from each book
    sample_sentences = [
        "Alexey Alexandrovitch cleared his throat and began to speak.",  # Anna Karenina
        "Alice was beginning to get very tired of sitting by her sister.",  # Alice in Wonderland
        "I am by birth a Genevese, and my family is one of the most distinguished of that republic.",  # Frankenstein
        "The family of the Julii was of Alban origin, and was settled at Rome from a very early period."  # Julius Caesar
    ]
    
    # Initialize model and tokenizer
    print("Loading model...")
    model = ConstructiveLearningModel()
    tokenizer = model.tokenizer
    
    # Set model to evaluation mode
    model.eval()
    
    print("Making predictions...")
    print()
    
    for i, sentence in enumerate(sample_sentences):
        # Tokenize the sentence
        inputs = tokenizer(
            sentence,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Sample {i+1}:")
        print(f"  Sentence: {sentence}")
        print(f"  Predicted book: {class_names[str(predicted_class)]}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  All probabilities:")
        for j, prob in enumerate(probabilities[0]):
            print(f"    {class_names[str(j)]}: {prob:.3f}")
        print()

if __name__ == "__main__":
    test_prediction() 