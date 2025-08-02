# Multi-Label Classification Training Pipeline

This directory contains a clean, organized pipeline for training and optimizing multi-label classification models for book classification.

## Overview

The pipeline trains three different model architectures:
- **Random Forest**: Ensemble method using multiple decision trees
- **Logistic Regression**: Linear model with regularization
- **SVM (Support Vector Machine)**: Kernel-based classification

## Files Structure

```
├── train_multi_label_models.py      # Train models with default parameters
├── optimize_hyperparameters.py      # Hyperparameter optimization
├── evaluate_optimized_models.py     # Evaluate and compare models
├── run_training_pipeline.py         # Main pipeline runner
└── TRAINING_PIPELINE_README.md      # This file
```

## Quick Start

Run the complete pipeline:
```bash
python run_training_pipeline.py
```

Or run individual steps:
```bash
# Step 1: Train with default parameters
python train_multi_label_models.py

# Step 2: Optimize hyperparameters
python optimize_hyperparameters.py

# Step 3: Evaluate optimized models
python evaluate_optimized_models.py
```

## Pipeline Steps

### 1. Training with Default Parameters (`train_multi_label_models.py`)

- Loads the semantic augmented dataset
- Trains Random Forest, Logistic Regression, and SVM models
- Uses default hyperparameters for each model
- Saves models and results to `trained_models/` directory

**Output:**
- Trained models with default parameters
- Performance metrics for each model
- Training summary

### 2. Hyperparameter Optimization (`optimize_hyperparameters.py`)

- Performs grid search optimization for each model type
- Uses 3-fold cross-validation
- Optimizes F1-score (weighted average)
- Saves best models and parameters

**Random Forest Parameters:**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]

**Logistic Regression Parameters:**
- `C`: [0.1, 1.0, 10.0, 100.0]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear', 'saga']
- `max_iter`: [1000, 2000]

**SVM Parameters:**
- `C`: [0.1, 1.0, 10.0, 100.0]
- `kernel`: ['linear', 'rbf', 'poly']
- `gamma`: ['scale', 'auto', 0.001, 0.01, 0.1]

**Output:**
- Best hyperparameters for each model
- Optimized models saved to `optimization_results/best_models/`
- Optimization results saved to `optimization_results/`

### 3. Model Evaluation (`evaluate_optimized_models.py`)

- Loads optimized models
- Evaluates on test set
- Compares performance across all models
- Creates visualizations

**Metrics Calculated:**
- Accuracy
- Hamming Loss
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Per-book metrics
- Average predictions per sample

**Output:**
- Detailed evaluation results
- Model comparison tables
- Performance visualizations
- Results saved to `evaluation_results/`

## Output Directories

### `trained_models/`
- Models trained with default parameters
- Training results and metrics
- Training summary

### `optimization_results/`
- Best hyperparameters for each model
- Optimized models
- Cross-validation scores
- Best model information

### `evaluation_results/`
- Detailed evaluation metrics
- Model comparison data
- Performance visualizations
- Plots directory with charts

## Dataset Requirements

The pipeline expects the semantic augmented dataset at:
```
data/semantic_augmented/semantic_augmented_dataset.csv
```

The dataset should contain:
- Binary features for each book (columns starting with 'book_')
- Labels for each book (same columns as features)
- ~31,761 samples (based on current dataset)

## Model Architecture

All models use `MultiOutputClassifier` wrapper to handle multi-label classification:

```python
# Example for Random Forest
base_model = RandomForestClassifier(**params)
model = MultiOutputClassifier(base_model)
```

This approach trains a separate classifier for each book label.

## Performance Metrics

### Primary Metrics
- **F1-Score (Weighted)**: Primary optimization target
- **Accuracy**: Overall classification accuracy
- **Hamming Loss**: Average fraction of wrong labels

### Secondary Metrics
- **Precision/Recall**: Per-book and weighted averages
- **Average Predictions**: Number of books predicted per sample
- **Confidence Scores**: Model prediction confidence

## Visualization

The evaluation script creates several visualizations:
- Model performance comparison bar charts
- Hamming loss comparison
- Average predictions per sample
- Per-book performance analysis

## Best Practices

1. **Data Preparation**: Ensure the semantic augmented dataset is properly formatted
2. **Memory Management**: Large datasets may require significant RAM
3. **Parallel Processing**: Models use `n_jobs=-1` for parallel training
4. **Reproducibility**: All scripts use `random_state=42` for consistency

## Troubleshooting

### Common Issues

1. **Dataset Not Found**: Ensure `data/semantic_augmented/semantic_augmented_dataset.csv` exists
2. **Memory Errors**: Reduce dataset size or use smaller parameter grids
3. **SVM Convergence**: Increase `max_iter` or reduce parameter space
4. **Long Training Times**: SVM optimization can be slow with large datasets

### Performance Tips

1. **Use Default Parameters First**: Run `train_multi_label_models.py` to get baseline performance
2. **Start with Smaller Grids**: Modify parameter grids in `optimize_hyperparameters.py` for faster optimization
3. **Monitor Memory Usage**: Large datasets may require system monitoring
4. **Save Intermediate Results**: All scripts save results automatically

## Next Steps

After running the pipeline:

1. **Analyze Results**: Review evaluation results and visualizations
2. **Select Best Model**: Choose the best performing model for your use case
3. **Further Optimization**: Modify parameter grids based on initial results
4. **Model Deployment**: Use the best model for inference

## Dependencies

Required packages (see `requirements.txt`):
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- logging
- pickle
- json 