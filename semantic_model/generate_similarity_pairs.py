#!/usr/bin/env python3
"""
Create similarity test pairs using book-based prompts.
This is a preliminary stage that generates test pairs once and saves them for reuse.
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityPairGenerator:
    """Generate similarity test pairs using book-based prompts."""
    
    def __init__(self, config_path: str = "configs/config.yaml", 
                 similar_sentences_per_book: int = 10, 
                 total_dissimilar_pairs: int = 40):
        """Initialize the pair generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.output_path = self.data_dir / "generated_similarity_pairs.json"
        
        # Generation parameters
        self.similar_sentences_per_book = similar_sentences_per_book
        self.total_dissimilar_pairs = total_dissimilar_pairs
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load the deduplicated corpus data."""
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load deduplicated corpus
        with open(self.data_dir / "corpus_deduplicated.json", 'r') as f:
            self.corpus = json.load(f)
        
        logger.info(f"Loaded corpus with {len(self.corpus)} books")
        logger.info(f"Books: {self.metadata['books']}")
        
        # Count total sentences
        total_sentences = sum(len(book_data['sentences']) for book_data in self.corpus.values())
        logger.info(f"Total sentences in corpus: {total_sentences}")
    
    def check_ollama_status(self):
        """Check if Ollama is running and working properly."""
        logger.info("Checking Ollama status...")
        
        try:
            import requests
            
            # Check if Ollama service is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama service returned status {response.status_code}")
            
            # Check if llama2:7b model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if 'llama2:7b' not in model_names:
                logger.warning("llama2:7b model not found. Available models:")
                for model in models:
                    logger.warning(f"  - {model['name']}")
                raise Exception("Required model 'llama2:7b' not found. Please run 'ollama pull llama2:7b'")
            
            # Test a simple generation to ensure it works
            test_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2:7b",
                    "prompt": "Say 'Hello, Ollama is working!'",
                    "stream": False
                },
                timeout=30
            )
            
            if test_response.status_code != 200:
                raise Exception(f"Ollama generation test failed with status {test_response.status_code}")
            
            logger.info("✅ Ollama is running and working properly")
            return True
            
        except requests.exceptions.ConnectionError:
            raise Exception(
                "Cannot connect to Ollama service. Please ensure Ollama is running:\n"
                "1. Start Ollama: ollama serve\n"
                "2. Or install if not installed: curl -fsSL https://ollama.ai/install.sh | sh"
            )
        except Exception as e:
            raise Exception(f"Ollama check failed: {e}")
    
    def get_book_samples(self) -> Dict[str, List[str]]:
        """Get sample sentences from each book."""
        book_samples = {}
        for book_name in self.metadata['books']:
            if book_name in self.corpus:
                book_sentences = self.corpus[book_name]['sentences']
                
                # Sample 1000 sentences per book randomly (or all if fewer)
                book_samples[book_name] = random.sample(book_sentences, min(1000, len(book_sentences)))
            else:
                logger.warning(f"Book '{book_name}' not found in corpus")
        
        logger.info(f"Sampled sentences from {len(book_samples)} books")
        for book_name, sentences in book_samples.items():
            logger.info(f"  {book_name}: {len(sentences)} sentences")
        
        return book_samples
    
    def create_similar_prompts_with_samples(self, book_samples: Dict[str, List[str]]) -> List[Dict]:
        """Create prompts for similar pairs using sampled sentences from books."""
        prompts = []
        
        for book_name, sentences in book_samples.items():
            if len(sentences) < self.similar_sentences_per_book:
                continue
                
            # Sample sentences from this book
            sampled_sentences = random.sample(sentences, self.similar_sentences_per_book)
            
            # Create one prompt per sentence
            for sentence in sampled_sentences:
                prompt = f"""
Write one sentence that is semantically similar to this sentence from "{book_name}":

"{sentence}"

Write only the similar sentence, no explanations or extra text.
"""
                
                prompts.append({
                    'prompt': prompt,
                    'topic': f"similar to {book_name}",
                    'source_book': book_name,
                    'original_sentence': sentence
                })
        
        return prompts
    
    def sample_dissimilar_pairs(self, book_samples: Dict[str, List[str]]) -> List[Dict]:
        """Create dissimilar pairs by randomly sampling sentences from different books."""
        prompts = []
        
        book_names = list(book_samples.keys())
        
        logger.info(f"Generating {self.total_dissimilar_pairs} dissimilar pairs...")
        
        # Generate pairs by sampling from different books
        for i in range(self.total_dissimilar_pairs):
            # Randomly select two different books
            book1, book2 = random.sample(book_names, 2)
            
            # Sample one sentence from each book
            sentence1 = random.choice(book_samples[book1])
            sentence2 = random.choice(book_samples[book2])
            
            prompts.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'book1': book1,
                'book2': book2,
                'type': 'dissimilar'
            })
        
        logger.info(f"Created {len(prompts)} dissimilar pairs")
        return prompts
    
    def _call_ai_model(self, prompt: str) -> List[Dict]:
        """Call AI model to generate sentence pairs."""
        # Try to use local model first (Ollama)
        try:
            return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            # Fallback to OpenAI API if available
            try:
                return self._call_openai(prompt)
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                raise Exception(
                    "Both Ollama and OpenAI failed to generate sentence pairs. "
                    "Please check your Ollama setup or OpenAI API configuration. "
                    "The script cannot proceed without proper AI generation."
                )
    
    def _call_ollama(self, prompt: str) -> List[Dict]:
        """Call Ollama local model."""
        try:
            import requests
            
            # Format prompt for Ollama
            formatted_prompt = f"""
{prompt}

Please respond with exactly 5 pairs in JSON format:
[
  {{"sentence1": "first sentence", "sentence2": "similar/dissimilar sentence"}},
  ...
]
"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2:7b",
                    "prompt": formatted_prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['response']
                
                # Try multiple patterns to extract JSON
                import re
                
                # Pattern 1: Look for JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                # Pattern 2: Look for individual JSON objects
                json_objects = re.findall(r'\{[^}]+\}', response_text)
                if len(json_objects) >= 5:
                    try:
                        pairs = []
                        for obj in json_objects[:5]:
                            pairs.append(json.loads(obj))
                        return pairs
                    except json.JSONDecodeError:
                        pass
                
                # Pattern 3: Try to extract sentence pairs from text
                logger.warning(f"Could not parse JSON from Ollama response: {response_text[:200]}...")
                raise Exception("Failed to parse JSON from Ollama response")
            
            raise Exception(f"Ollama API returned status {response.status_code}")
            
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> List[Dict]:
        """Call OpenAI API."""
        try:
            import openai
            
            # Check if OpenAI API key is available
            if not hasattr(openai, 'api_key') or not openai.api_key:
                raise Exception("OpenAI API key not configured")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates sentence pairs for semantic similarity testing."},
                    {"role": "user", "content": f"{prompt}\n\nRespond with exactly 5 pairs in JSON format."}
                ],
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            raise Exception("Failed to parse OpenAI response")
            
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            raise    

    def _call_ai_model_for_sentence_generation(self, prompt: str) -> str:
        """Call AI model to generate a single similar sentence."""
        try:
            response = self._call_ollama_sentence(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama sentence generation failed: {e}")
            try:
                response = self._call_openai_sentence(prompt)
                return response
            except Exception as e:
                logger.error(f"OpenAI sentence generation failed: {e}")
                raise Exception(
                    "Both Ollama and OpenAI failed to generate sentences. "
                    "Please check your AI model setup."
                )
    
    def _call_ollama_sentence(self, prompt: str) -> str:
        """Call Ollama for simple sentence generation."""
        try:
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2:7b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['response'].strip()
                
                # Clean up the response - remove any extra text
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Similar sentence:') and not line.startswith('Original:') and not line.startswith('Sure!') and not line.startswith('Certainly!') and not line.startswith('Here is'):
                        # Remove quotes if present
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]
                        return line
                
                # If no clean line found, return the whole response
                return response_text
            
            raise Exception(f"Ollama API returned status {response.status_code}")
            
        except Exception as e:
            logger.error(f"Ollama sentence call failed: {e}")
            raise
    
    def _call_openai_sentence(self, prompt: str) -> str:
        """Call OpenAI for simple sentence generation."""
        try:
            import openai
            
            if not hasattr(openai, 'api_key') or not openai.api_key:
                raise Exception("OpenAI API key not configured")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates semantically similar sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            content = response.choices[0].message.content.strip()
            return content
                
        except Exception as e:
            logger.warning(f"OpenAI sentence call failed: {e}")
            raise

    def generate_similar_pairs(self, book_samples: Dict[str, List[str]]) -> List[Dict]:
        """Generate semantically similar sentence pairs using sampled sentences from books."""
        similar_pairs = []
        
        # Create prompts using sampled sentences from books
        prompts = self.create_similar_prompts_with_samples(book_samples)
        
        logger.info(f"Generating similar pairs for {len(prompts)} sentences...")
        
        for i, prompt_data in enumerate(prompts, 1):
            logger.info(f"  [{i}/{len(prompts)}] Generating similar sentence for '{prompt_data['source_book']}'...")
            
            try:
                # Use local model or API to generate similar sentence
                similar_sentence = self._call_ai_model_for_sentence_generation(prompt_data['prompt'])
                logger.info(f"    ✅ Generated similar sentence for '{prompt_data['source_book']}'")
                
                similar_pairs.append({
                    'sentence1': prompt_data['original_sentence'],
                    'sentence2': similar_sentence,
                    'similarity': 1.0,  # Should be high
                    'type': 'similar',
                    'topic': prompt_data['topic'],
                    'source_book': prompt_data['source_book']
                })
            except Exception as e:
                logger.error(f"    ❌ Failed to generate similar sentence for '{prompt_data['source_book']}': {e}")
                # Continue with next sentence instead of failing completely
                continue
        
        logger.info(f"✅ Successfully generated {len(similar_pairs)} similar pairs")
        return similar_pairs
    
    def generate_dissimilar_pairs(self, book_samples: Dict[str, List[str]]) -> List[Dict]:
        """Generate dissimilar sentence pairs by randomly sampling from different books."""
        dissimilar_pairs = []
        
        # Create dissimilar pairs by sampling from different books
        pairs = self.sample_dissimilar_pairs(book_samples)
        
        logger.info(f"Generated {len(pairs)} dissimilar pairs from different books...")
        
        for i, prompt_data in enumerate(pairs, 1):
            logger.info(f"  [{i}/{len(pairs)}] Pair from '{prompt_data['book1']}' and '{prompt_data['book2']}'...")
            
            # Add the dissimilar pair directly
            dissimilar_pairs.append({
                'sentence1': prompt_data['sentence1'],
                'sentence2': prompt_data['sentence2'],
                'similarity': 0.0,  # Should be low
                'type': 'dissimilar',
                'book1': prompt_data['book1'],
                'book2': prompt_data['book2']
            })
            logger.info(f"    ✅ Added dissimilar pair ({len(dissimilar_pairs)}/{self.total_dissimilar_pairs})")
        
        logger.info(f"✅ Successfully generated {len(dissimilar_pairs)} dissimilar pairs")
        return dissimilar_pairs
    
    def create_similarity_test_pairs(self) -> List[Dict]:
        """Create test pairs for similarity evaluation using generative AI."""
        logger.info("Creating similarity test pairs using generative AI...")
        
        # Check if test pairs already exist
        if self.output_path.exists():
            logger.info("Loading existing test pairs...")
            with open(self.output_path, 'r') as f:
                return json.load(f)
        
        # Check Ollama status before proceeding
        self.check_ollama_status()
        
        # Generate test pairs using generative AI
        test_pairs = self._generate_test_pairs_with_ai()
        
        # Save test pairs
        with open(self.output_path, 'w') as f:
            json.dump(test_pairs, f, indent=2)
        
        logger.info(f"Created {len(test_pairs)} test pairs")
        return test_pairs
    
    def _generate_test_pairs_with_ai(self) -> List[Dict]:
        """Generate test pairs using generative AI."""
        logger.info("Generating test pairs with AI...")
        
        # Get sample sentences from each book
        book_samples = self.get_book_samples()
        
        logger.info("=" * 50)
        logger.info("STEP 1: Generating similar pairs")
        logger.info("=" * 50)
        
        # Create prompt for generating similar pairs
        similar_pairs = self.generate_similar_pairs(book_samples)
        
        logger.info("=" * 50)
        logger.info("STEP 2: Generating dissimilar pairs")
        logger.info("=" * 50)
        
        # Create prompt for generating dissimilar pairs
        dissimilar_pairs = self.generate_dissimilar_pairs(book_samples)
        
        # Combine all pairs
        all_pairs = similar_pairs + dissimilar_pairs
        
        logger.info("=" * 50)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total pairs generated: {len(all_pairs)}")
        logger.info(f"  - Similar pairs: {len(similar_pairs)}")
        logger.info(f"  - Dissimilar pairs: {len(dissimilar_pairs)}")
        
        return all_pairs

def check_ollama_only():
    """Check only Ollama status without creating pairs."""
    generator = SimilarityPairGenerator()
    generator.check_ollama_status()
    print("✅ Ollama is ready for use!")

def create_similarity_pairs(config_path: str = "configs/config.yaml", 
                          similar_sentences_per_book: int = 10,
                          total_dissimilar_pairs: int = 40):
    """Create similarity test pairs."""
    generator = SimilarityPairGenerator(
        config_path=config_path,
        similar_sentences_per_book=similar_sentences_per_book,
        total_dissimilar_pairs=total_dissimilar_pairs
    )
    test_pairs = generator.create_similarity_test_pairs()
    
    print(f"\n✅ Similarity test pairs created successfully!")
    print(f"📁 Saved to: {generator.output_path}")
    print(f"📊 Total pairs: {len(test_pairs)}")
    
    # Count by type
    similar_count = len([p for p in test_pairs if p['type'] == 'similar'])
    dissimilar_count = len([p for p in test_pairs if p['type'] == 'dissimilar'])
    print(f"   Similar pairs: {similar_count}")
    print(f"   Dissimilar pairs: {dissimilar_count}")
    
    return test_pairs

def main():
    """Main function to create similarity pairs."""
    parser = argparse.ArgumentParser(description="Create similarity test pairs")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force regeneration of test pairs")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check Ollama status without creating pairs")
    parser.add_argument("--similar-sentences-per-book", type=int, default=10,
                       help="Number of sentences to sample per book for similar pairs (default: 10)")
    parser.add_argument("--total-dissimilar-pairs", type=int, default=40,
                       help="Total number of dissimilar pairs to generate (default: 40)")
    
    args = parser.parse_args()
    
    if args.check_only:
        check_ollama_only()
        return
    
    # If force flag is set, remove existing file
    if args.force:
        output_path = Path("data/generated_similarity_pairs.json")
        if output_path.exists():
            output_path.unlink()
            print("🗑️  Removed existing test pairs file")
    
    test_pairs = create_similarity_pairs(
        config_path=args.config,
        similar_sentences_per_book=args.similar_sentences_per_book,
        total_dissimilar_pairs=args.total_dissimilar_pairs
    )
    
    if not test_pairs:
        sys.exit(1)

if __name__ == "__main__":
    main() 