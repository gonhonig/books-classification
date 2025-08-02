# Individual Book Models

This folder contains the implementation and results for the individual book classification approach, where separate binary classifiers are trained for each book.

## Approach

The individual book models approach trains separate binary classifiers for each book:
- **Anna Karenina** classifier
- **Wuthering Heights** classifier  
- **Frankenstein** classifier
- **The Adventures of Alice in Wonderland** classifier

Each model is trained to predict whether a sentence belongs to its specific book or not, using the same balanced dataset with aligned embeddings.

## Files

### Training Script
- `train_models.py` - Main training script for individual book models

### Model Files
- `anna_karenina_best_model.pth` - Trained model for Anna Karenina
- `wuthering_heights_best_model.pth` - Trained model for Wuthering Heights
- `frankenstein_best_model.pth` - Trained model for Frankenstein
- `the_adventures_of_alice_in_wonderland_best_model.pth` - Trained model for Alice in Wonderland

### Results and Analysis
- `results.json` - Training results and metrics for all models
- `model_comparison.png` - Visualization comparing performance across all models
- `test_examples.json` - Sample predictions on test data
- `test_examples_report.md` - Detailed report of test examples
- `per_book_detailed_metrics.json` - Detailed per-book single-label vs multi-label metrics
- `per_book_detailed_metrics_report.md` - Comprehensive performance analysis report

## Usage

### Training Models
```bash
python individual_book_models/train_models.py
```

This will:
1. Load the pre-existing dataset splits from `data/dataset`
2. Load aligned embeddings from `data/embeddings_cache_aligned_*.npz`
3. Train separate binary classifiers for each book
4. Save trained models, results, and analysis files

### Key Features

#### Model Architecture
- **Input**: 384-dimensional sentence embeddings
- **Hidden Layers**: [256, 128, 64] with ReLU activation
- **Regularization**: BatchNorm and Dropout (0.3)
- **Output**: Sigmoid activation for binary classification

#### Training Process
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Binary Cross-Entropy Loss
- **Scheduling**: ReduceLROnPlateau with patience=5
- **Early Stopping**: Patience=15 epochs
- **Validation**: Uses pre-existing validation split

#### Performance Analysis
- **Overall Metrics**: Accuracy, Precision, Recall, F1-score
- **Multi-label Analysis**: Performance on sentences belonging to multiple books
- **Single-label Analysis**: Performance on sentences belonging to single books
- **Per-book Comparison**: Detailed breakdown for each book

## Results Summary

### Overall Performance (Latest Run)
| Book | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Anna Karenina | 87.4% | 89.6% | 84.8% | 87.1% |
| Wuthering Heights | 84.1% | 81.9% | 86.9% | 84.3% |
| Frankenstein | 86.3% | 87.6% | 80.9% | 84.1% |
| Alice in Wonderland | 88.7% | 78.4% | 80.5% | 79.4% |

### Multi-label vs Single-label Performance
- **Anna Karenina**: Multi-label F1 91.2% vs Single-label F1 41.3% (+49.9%)
- **Wuthering Heights**: Multi-label F1 89.7% vs Single-label F1 58.2% (+31.4%)
- **Frankenstein**: Multi-label F1 69.9% vs Single-label F1 91.7% (-21.8%)
- **Alice in Wonderland**: Multi-label F1 75.1% vs Single-label F1 86.2% (-11.1%)

## Key Insights

1. **Anna Karenina & Wuthering Heights** excel at multi-label classification, suggesting strong semantic overlap
2. **Frankenstein & Alice in Wonderland** perform better on single-label classification, indicating more distinct content
3. **All models achieve 80%+ overall accuracy**, showing excellent general performance
4. **Multi-label scenarios are more challenging** for most books, but Anna Karenina and Wuthering Heights handle them exceptionally well

## Data Requirements

- `data/dataset/` - HuggingFace DatasetDict with train/validation/test splits
- `data/embeddings_cache_aligned_*.npz` - Aligned embeddings for the dataset splits

## Dependencies

- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- datasets (HuggingFace)
- sentence-transformers (for embeddings)

## Notes

- Models are saved in PyTorch format (.pth)
- All results are saved in JSON format for easy analysis
- Detailed reports are generated in Markdown format
- The approach uses perfectly aligned embeddings to avoid data misalignment issues 