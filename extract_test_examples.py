"""
Extract Test Examples
Extract examples of single-label and multi-label sentences from test data,
showing original book, test labels, and predictions from each model.
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

class TestExampleExtractor:
    """Extract test examples showing model performance."""
    
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
    
    def identify_sentence_types(self):
        """Identify single-label and multi-label sentences."""
        single_label_indices = []
        multi_label_indices = []
        
        for i in range(len(self.df)):
            # Count how many books this sentence belongs to
            book_labels_for_sentence = [
                self.book_labels['book_1'][i],
                self.book_labels['book_2'][i], 
                self.book_labels['book_3'][i],
                self.book_labels['book_4'][i]
            ]
            
            if sum(book_labels_for_sentence) == 1:
                single_label_indices.append(i)
            elif sum(book_labels_for_sentence) > 1:
                multi_label_indices.append(i)
        
        logger.info(f"Found {len(single_label_indices)} single-label sentences")
        logger.info(f"Found {len(multi_label_indices)} multi-label sentences")
        
        return single_label_indices, multi_label_indices
    
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
    
    def extract_examples(self, num_single_label=10, num_multi_label=10):
        """Extract examples of single-label and multi-label sentences."""
        logger.info("Extracting test examples...")
        
        # Load models
        self.load_models()
        
        # Identify sentence types
        single_label_indices, multi_label_indices = self.identify_sentence_types()
        
        examples = {
            'single_label_examples': [],
            'multi_label_examples': []
        }
        
        # Extract single-label examples
        logger.info(f"Extracting {num_single_label} single-label examples...")
        for i in range(min(num_single_label, len(single_label_indices))):
            idx = single_label_indices[i]
            
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
        
        # Extract multi-label examples
        logger.info(f"Extracting {num_multi_label} multi-label examples...")
        for i in range(min(num_multi_label, len(multi_label_indices))):
            idx = multi_label_indices[i]
            
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
        report = "# Test Examples Analysis\n\n"
        
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
    logger.info("Starting test example extraction...")
    
    # Create extractor
    extractor = TestExampleExtractor()
    
    # Extract examples
    examples = extractor.extract_examples(num_single_label=15, num_multi_label=15)
    
    # Calculate accuracy summary
    summary = extractor.calculate_accuracy_summary(examples)
    
    # Create detailed report
    report = extractor.create_detailed_report(examples)
    
    # Add summary to report
    summary_section = f"""
## Accuracy Summary

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
    with open('models/test_examples_report.md', 'w') as f:
        f.write(report)
    
    # Save examples as JSON for further analysis
    with open('models/test_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info("Test examples extracted and saved!")
    logger.info(f"Single-label accuracy: {summary['single_label_accuracy']:.3f}")
    logger.info(f"Multi-label accuracy: {summary['multi_label_accuracy']:.3f}")
    
    print("\n" + "="*50)
    print("TEST EXAMPLES EXTRACTION COMPLETED")
    print("="*50)
    print(f"Single-label accuracy: {summary['single_label_accuracy']:.3f}")
    print(f"Multi-label accuracy: {summary['multi_label_accuracy']:.3f}")
    print(f"Files saved:")
    print("- models/test_examples_report.md")
    print("- models/test_examples.json")

if __name__ == "__main__":
    main() 