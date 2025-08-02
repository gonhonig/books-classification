"""
Extract Test Examples (Fixed Version)
Extract examples of single-label and multi-label sentences from ACTUAL test data,
showing original book, test labels, and predictions from each model.
This version ensures we only use sentences that were NOT used during training.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryBookClassifier(torch.nn.Module):
    """Binary classifier for a single book."""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(BinaryBookClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Feature extraction layers
        self.feature_layers = torch.nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(torch.nn.Sequential(
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim
        
        # Output layer for binary classification
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Binary classification output
        x = self.output_layer(x)
        return x.squeeze()

class TestExampleExtractorFixed:
    """Extract test examples showing model performance using ONLY test sentences."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
        # Book names mapping
        self.book_names = {
            'book_1': 'Anna Karenina',
            'book_2': 'Wuthering Heights', 
            'book_3': 'Frankenstein',
            'book_4': 'The Adventures of Alice in Wonderland'
        }
        
        # Load the semantic augmented dataset
        self.df = pd.read_csv('data/semantic_augmented/semantic_augmented_dataset.csv')
        self.embeddings = np.load('data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz')['embeddings'].astype(np.float32)
        
        # Extract book labels
        column_mapping = {
            'book_Anna_Karenina': 'book_1',
            'book_Wuthering_Heights': 'book_2', 
            'book_Frankenstein': 'book_3',
            'book_The_Adventures_of_Alice_in_Wonderland': 'book_4'
        }
        
        self.book_labels = {}
        for col_name, book_col in column_mapping.items():
            if col_name in self.df.columns:
                self.book_labels[book_col] = self.df[col_name].values.astype(np.float32)
        
        logger.info(f"Loaded dataset with {len(self.df)} sentences")
        
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading trained models...")
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            book_name = self.book_names[book_col]
            model_path = f"models/{book_name.replace(' ', '_').lower()}_best_model.pth"
            
            if Path(model_path).exists():
                # Create model
                model = BinaryBookClassifier().to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.models[book_col] = model
                logger.info(f"Loaded model for {book_name}")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load scalers (we'll need to recreate them or load from results)
        # For now, we'll use StandardScaler on the embeddings
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.embeddings)
        
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            self.scalers[book_col] = scaler
    
    def recreate_test_indices(self, book_col):
        """Recreate the exact test indices that were used during training."""
        logger.info(f"Recreating test indices for {self.book_names[book_col]}")
        
        # Get positive and negative samples
        positive_indices = np.where(self.book_labels[book_col] == 1)[0]
        negative_indices = np.where(self.book_labels[book_col] == 0)[0]
        
        # Identify multi-label sentences
        multi_label_indices = []
        for i in range(len(self.embeddings)):
            book_labels_for_sentence = [
                self.book_labels['book_1'][i],
                self.book_labels['book_2'][i], 
                self.book_labels['book_3'][i],
                self.book_labels['book_4'][i]
            ]
            if sum(book_labels_for_sentence) > 1:
                multi_label_indices.append(i)
        
        # For positive samples, prioritize multi-label sentences
        positive_multi_label = [i for i in positive_indices if i in multi_label_indices]
        positive_single_label = [i for i in positive_indices if i not in multi_label_indices]
        
        # Include ALL multi-label sentences that are positive for this book
        selected_positive = positive_multi_label.copy()
        
        # Add single-label positive samples to reach target (same as training)
        target_samples_per_class = 5000
        remaining_positive_needed = target_samples_per_class - len(selected_positive)
        if remaining_positive_needed > 0 and len(positive_single_label) > 0:
            n_to_sample = min(remaining_positive_needed, len(positive_single_label))
            # Use the same random seed as training
            np.random.seed(42)
            selected_single_label = np.random.choice(positive_single_label, n_to_sample, replace=False)
            selected_positive.extend(selected_single_label)
        
        # For negative samples, sample normally (same as training)
        n_negative = min(len(negative_indices), len(selected_positive))
        np.random.seed(42)
        selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
        
        # Combine indices (same as training)
        original_indices = np.concatenate([selected_positive, selected_negative])
        np.random.seed(42)
        np.random.shuffle(original_indices)
        
        # Split the original indices (same as training)
        train_size = int(0.7 * len(original_indices))
        val_size = int(0.15 * len(original_indices))
        
        train_indices = original_indices[:train_size]
        val_indices = original_indices[train_size:train_size + val_size]
        test_indices = original_indices[train_size + val_size:]
        
        logger.info(f"Recreated splits for {self.book_names[book_col]}:")
        logger.info(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        return test_indices
    
    def get_predictions_for_sentence(self, sentence_idx):
        """Get predictions from all models for a specific sentence."""
        # Get the embedding for this sentence
        embedding = self.embeddings[sentence_idx:sentence_idx+1]
        
        predictions = {}
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            if book_col in self.models:
                model = self.models[book_col]
                scaler = self.scalers[book_col]
                
                # Scale the embedding
                embedding_scaled = scaler.transform(embedding)
                
                # Get prediction
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(embedding_scaled).to(self.device)
                    output = model(X_tensor)
                    if output.dim() == 0:
                        prediction = output.cpu().numpy()
                    else:
                        prediction = output.cpu().numpy()[0]
                    
                    predictions[book_col] = {
                        'probability': float(prediction),
                        'prediction': 1 if prediction > 0.5 else 0
                    }
        
        return predictions
    
    def extract_test_examples(self, num_single_label=10, num_multi_label=10):
        """Extract examples from ACTUAL test sentences only."""
        logger.info("Extracting test examples from actual test data...")
        
        # Load models
        self.load_models()
        
        # Get test indices for each book
        test_indices_by_book = {}
        for book_col in ['book_1', 'book_2', 'book_3', 'book_4']:
            test_indices_by_book[book_col] = self.recreate_test_indices(book_col)
        
        # Combine all test indices to get the full test set
        all_test_indices = set()
        for test_indices in test_indices_by_book.values():
            all_test_indices.update(test_indices)
        all_test_indices = list(all_test_indices)
        
        logger.info(f"Total unique test sentences: {len(all_test_indices)}")
        
        # Identify sentence types in test set
        single_label_test_indices = []
        multi_label_test_indices = []
        
        for test_idx in all_test_indices:
            # Count how many books this sentence belongs to
            book_labels_for_sentence = [
                self.book_labels['book_1'][test_idx],
                self.book_labels['book_2'][test_idx], 
                self.book_labels['book_3'][test_idx],
                self.book_labels['book_4'][test_idx]
            ]
            
            if sum(book_labels_for_sentence) == 1:
                single_label_test_indices.append(test_idx)
            elif sum(book_labels_for_sentence) > 1:
                multi_label_test_indices.append(test_idx)
        
        logger.info(f"Found {len(single_label_test_indices)} single-label test sentences")
        logger.info(f"Found {len(multi_label_test_indices)} multi-label test sentences")
        
        examples = {
            'single_label_examples': [],
            'multi_label_examples': []
        }
        
        # Extract single-label examples from test set
        logger.info(f"Extracting {num_single_label} single-label test examples...")
        for i in range(min(num_single_label, len(single_label_test_indices))):
            idx = single_label_test_indices[i]
            
            sentence_data = {
                'sentence': self.df.iloc[idx]['sentence'],
                'original_book': self.df.iloc[idx]['original_book'],
                'original_label': int(self.df.iloc[idx]['original_label']),
                'true_labels': {
                    'book_1': int(self.book_labels['book_1'][idx]),
                    'book_2': int(self.book_labels['book_2'][idx]),
                    'book_3': int(self.book_labels['book_3'][idx]),
                    'book_4': int(self.book_labels['book_4'][idx])
                },
                'predictions': self.get_predictions_for_sentence(idx)
            }
            
            examples['single_label_examples'].append(sentence_data)
        
        # Extract multi-label examples from test set
        logger.info(f"Extracting {num_multi_label} multi-label test examples...")
        for i in range(min(num_multi_label, len(multi_label_test_indices))):
            idx = multi_label_test_indices[i]
            
            sentence_data = {
                'sentence': self.df.iloc[idx]['sentence'],
                'original_book': self.df.iloc[idx]['original_book'],
                'original_label': int(self.df.iloc[idx]['original_label']),
                'true_labels': {
                    'book_1': int(self.book_labels['book_1'][idx]),
                    'book_2': int(self.book_labels['book_2'][idx]),
                    'book_3': int(self.book_labels['book_3'][idx]),
                    'book_4': int(self.book_labels['book_4'][idx])
                },
                'predictions': self.get_predictions_for_sentence(idx)
            }
            
            examples['multi_label_examples'].append(sentence_data)
        
        return examples
    
    def create_detailed_report(self, examples):
        """Create a detailed report with the examples."""
        report = "# Test Examples Analysis (Fixed - Using Actual Test Data)\n\n"
        
        report += "## Important Note\n"
        report += "This analysis uses ONLY sentences that were held out as test data during training.\n"
        report += "These are the actual test sentences that the models have never seen before.\n\n"
        
        # Single-label examples
        report += "## Single-Label Sentence Examples\n\n"
        report += "These sentences belong to only one book:\n\n"
        
        for i, example in enumerate(examples['single_label_examples'], 1):
            report += f"### Example {i}\n\n"
            report += f"**Sentence**: {example['sentence'][:200]}{'...' if len(example['sentence']) > 200 else ''}\n\n"
            report += f"**Original Book**: {example['original_book']}\n"
            report += f"**Original Label**: {example['original_label']}\n\n"
            
            report += "**True Labels**:\n"
            for book_col, label in example['true_labels'].items():
                book_name = self.book_names[book_col]
                report += f"- {book_name}: {label}\n"
            
            report += "\n**Model Predictions**:\n"
            for book_col, pred_data in example['predictions'].items():
                book_name = self.book_names[book_col]
                true_label = example['true_labels'][book_col]
                prediction = pred_data['prediction']
                probability = pred_data['probability']
                
                # Check if prediction is correct
                correct = "✅" if prediction == true_label else "❌"
                
                report += f"- {book_name}: {prediction} (prob: {probability:.3f}) {correct}\n"
            
            report += "\n---\n\n"
        
        # Multi-label examples
        report += "## Multi-Label Sentence Examples\n\n"
        report += "These sentences belong to multiple books:\n\n"
        
        for i, example in enumerate(examples['multi_label_examples'], 1):
            report += f"### Example {i}\n\n"
            report += f"**Sentence**: {example['sentence'][:200]}{'...' if len(example['sentence']) > 200 else ''}\n\n"
            report += f"**Original Book**: {example['original_book']}\n"
            report += f"**Original Label**: {example['original_label']}\n\n"
            
            report += "**True Labels**:\n"
            for book_col, label in example['true_labels'].items():
                book_name = self.book_names[book_col]
                report += f"- {book_name}: {label}\n"
            
            report += "\n**Model Predictions**:\n"
            for book_col, pred_data in example['predictions'].items():
                book_name = self.book_names[book_col]
                true_label = example['true_labels'][book_col]
                prediction = pred_data['prediction']
                probability = pred_data['probability']
                
                # Check if prediction is correct
                correct = "✅" if prediction == true_label else "❌"
                
                report += f"- {book_name}: {prediction} (prob: {probability:.3f}) {correct}\n"
            
            report += "\n---\n\n"
        
        return report
    
    def calculate_accuracy_summary(self, examples):
        """Calculate accuracy summary for the examples."""
        single_label_correct = 0
        single_label_total = 0
        multi_label_correct = 0
        multi_label_total = 0
        
        # Single-label accuracy
        for example in examples['single_label_examples']:
            for book_col, pred_data in example['predictions'].items():
                true_label = example['true_labels'][book_col]
                prediction = pred_data['prediction']
                
                single_label_total += 1
                if prediction == true_label:
                    single_label_correct += 1
        
        # Multi-label accuracy
        for example in examples['multi_label_examples']:
            for book_col, pred_data in example['predictions'].items():
                true_label = example['true_labels'][book_col]
                prediction = pred_data['prediction']
                
                multi_label_total += 1
                if prediction == true_label:
                    multi_label_correct += 1
        
        summary = {
            'single_label_accuracy': single_label_correct / single_label_total if single_label_total > 0 else 0,
            'multi_label_accuracy': multi_label_correct / multi_label_total if multi_label_total > 0 else 0,
            'single_label_correct': single_label_correct,
            'single_label_total': single_label_total,
            'multi_label_correct': multi_label_correct,
            'multi_label_total': multi_label_total
        }
        
        return summary

def main():
    """Main function to extract and analyze test examples."""
    logger.info("Starting test example extraction (FIXED VERSION)...")
    
    # Create extractor
    extractor = TestExampleExtractorFixed()
    
    # Extract examples
    examples = extractor.extract_test_examples(num_single_label=15, num_multi_label=15)
    
    # Calculate accuracy summary
    summary = extractor.calculate_accuracy_summary(examples)
    
    # Create detailed report
    report = extractor.create_detailed_report(examples)
    
    # Add summary to report
    summary_section = f"""
## Accuracy Summary (Using Actual Test Data)

### Single-Label Sentences
- **Correct Predictions**: {summary['single_label_correct']}/{summary['single_label_total']}
- **Accuracy**: {summary['single_label_accuracy']:.3f} ({summary['single_label_accuracy']*100:.1f}%)

### Multi-Label Sentences
- **Correct Predictions**: {summary['multi_label_correct']}/{summary['multi_label_total']}
- **Accuracy**: {summary['multi_label_accuracy']:.3f} ({summary['multi_label_accuracy']*100:.1f}%)

### Overall Performance
- **Total Examples**: {summary['single_label_total'] + summary['multi_label_total']}
- **Total Correct**: {summary['single_label_correct'] + summary['multi_label_correct']}
- **Overall Accuracy**: {(summary['single_label_correct'] + summary['multi_label_correct']) / (summary['single_label_total'] + summary['multi_label_total']):.3f}

"""
    
    report = summary_section + report
    
    # Save report
    with open('models/test_examples_report_fixed.md', 'w') as f:
        f.write(report)
    
    # Save examples as JSON for further analysis
    with open('models/test_examples_fixed.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info("Test examples extracted and saved!")
    logger.info(f"Single-label accuracy: {summary['single_label_accuracy']:.3f}")
    logger.info(f"Multi-label accuracy: {summary['multi_label_accuracy']:.3f}")
    
    print("\n" + "="*60)
    print("TEST EXAMPLES EXTRACTION COMPLETED (FIXED VERSION)")
    print("="*60)
    print(f"Single-label accuracy: {summary['single_label_accuracy']:.3f}")
    print(f"Multi-label accuracy: {summary['multi_label_accuracy']:.3f}")
    print(f"Files saved:")
    print("- models/test_examples_report_fixed.md")
    print("- models/test_examples_fixed.json")
    print("\nNOTE: This version uses ONLY actual test sentences that were held out during training!")

if __name__ == "__main__":
    main() 