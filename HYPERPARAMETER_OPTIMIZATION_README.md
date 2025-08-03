# Hyperparameter Optimization for Neural Networks

This module provides comprehensive hyperparameter optimization for both multi-label and per-book neural network models using Optuna.

## Features

- **Multi-label Model Optimization**: Optimizes hyperparameters for the unified multi-label neural network
- **Per-book Model Optimization**: Optimizes hyperparameters for individual binary classifiers
- **Comprehensive Parameter Search**: Architecture, training, and regularization parameters
- **Visualization**: Optimization history and parameter importance plots
- **Detailed Reporting**: Markdown reports with trial history and best parameters
- **Easy Integration**: Simple API to apply optimized parameters to training scripts

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Optimization

#### Multi-label Model Optimization
```bash
python run_hyperparameter_optimization.py --model_type multi_label --n_trials 30
```

#### Per-book Models Optimization
```bash
python run_hyperparameter_optimization.py --model_type per_book --n_trials 30
```

#### Specific Book Optimization
```bash
python run_hyperparameter_optimization.py --model_type specific_book --book_name "Anna Karenina" --n_trials 30
```

#### Both Model Types
```bash
python run_hyperparameter_optimization.py --model_type both --n_trials 30
```

#### Quick Mode (for testing)
```bash
python run_hyperparameter_optimization.py --model_type multi_label --quick
```

### 3. View Results

```bash
# List available optimization results
python utils/apply_optimized_params.py --list

# Get optimized parameters for a specific study
python utils/apply_optimized_params.py --study_name multi_label_optimization_20241201_143022
```

## Optimization Parameters

### Multi-label Model Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_layers` | 2-4 | Number of hidden layers |
| `hidden_dim_i` | 64-512 | Size of hidden layer i |
| `learning_rate` | 1e-5 to 1e-2 | Learning rate (log scale) |
| `batch_size` | [16, 32, 64, 128] | Batch size |
| `dropout_rate` | 0.1-0.5 | Dropout rate |
| `weight_decay` | 1e-6 to 1e-3 | Weight decay (log scale) |
| `epochs` | 50-200 | Maximum training epochs |
| `patience` | 10-30 | Early stopping patience |

### Per-book Model Parameters

Same parameter ranges as multi-label model, but optimized individually for each book.

## Output Structure

```
optimization_results/
├── multi_label_optimization_YYYYMMDD_HHMMSS/
│   ├── multi_label_optimization_YYYYMMDD_HHMMSS.pkl          # Optuna study
│   ├── multi_label_optimization_YYYYMMDD_HHMMSS_report.md     # Detailed report
│   ├── multi_label_optimization_YYYYMMDD_HHMMSS_best_params.json  # Best parameters
│   └── multi_label_optimization.png                          # Optimization plots
├── book_1_optimization_YYYYMMDD_HHMMSS/
│   ├── book_1_optimization_YYYYMMDD_HHMMSS.pkl
│   ├── book_1_optimization_YYYYMMDD_HHMMSS_report.md
│   ├── book_1_optimization_YYYYMMDD_HHMMSS_best_params.json
│   └── book_1_optimization.png
└── ... (similar for other books)
```

## Applying Optimized Parameters

### Method 1: Using the Utility Script

```bash
# Get optimized parameters
python utils/apply_optimized_params.py --study_name multi_label_optimization_20241201_143022
```

### Method 2: Direct Integration

```python
from utils.apply_optimized_params import load_optimized_params, apply_multi_label_params

# Load optimized parameters
params = load_optimized_params("multi_label_optimization_20241201_143022")

# Convert to training parameters
training_params = apply_multi_label_params(params)

# Use in training
trainer = NeuralNetworkTrainer()
trainer.load_data()
trainer.create_model(**training_params)
trainer.train_model(**training_params)
```

### Method 3: Specific Book Training

```python
# Train specific book with optimized parameters
python train_with_optimized_params.py --model_type specific_book --book_name "Anna Karenina" --study_name anna_karenina_optimization_20241201_143022
```

## Example Optimization Results

### Multi-label Model
```json
{
  "n_layers": 3,
  "hidden_dim_0": 256,
  "hidden_dim_1": 128,
  "hidden_dim_2": 64,
  "learning_rate": 0.001,
  "batch_size": 64,
  "dropout_rate": 0.3,
  "weight_decay": 1e-5,
  "epochs": 150,
  "patience": 20
}
```

### Per-book Model (Anna Karenina)
```json
{
  "n_layers": 3,
  "hidden_dim_0": 512,
  "hidden_dim_1": 256,
  "hidden_dim_2": 128,
  "learning_rate": 0.0005,
  "batch_size": 32,
  "dropout_rate": 0.4,
  "weight_decay": 1e-4,
  "epochs": 200,
  "patience": 25
}
```

## Advanced Usage

### Custom Optimization

```python
from utils.hyperparameter_optimizer import HyperparameterOptimizer

# Create custom optimizer
optimizer = HyperparameterOptimizer(
    model_type="multi_label",
    n_trials=100,
    timeout=7200,  # 2 hours
    study_name="custom_optimization"
)

# Load data
optimizer.load_data()

# Run optimization
study = optimizer.optimize_multi_label()

# Save results
optimizer.save_optimization_results(study, "multi_label")

# Create plots
optimizer.plot_optimization_history(study, "custom_optimization.png")
```

### Custom Parameter Ranges

Modify the `suggest_multi_label_hyperparameters` or `suggest_per_book_hyperparameters` methods in `utils/hyperparameter_optimizer.py` to customize parameter ranges.

## Performance Monitoring

The optimization process provides:

1. **Real-time Logging**: Progress updates during optimization
2. **Trial History**: Complete record of all trials
3. **Parameter Importance**: Which parameters most affect performance
4. **Visualization**: Optimization history and importance plots
5. **Detailed Reports**: Markdown reports with comprehensive analysis

## Tips for Best Results

1. **Start with Quick Mode**: Use `--quick` flag to test the setup
2. **Increase Trials**: More trials = better optimization (but longer runtime)
3. **Monitor Timeout**: Set appropriate timeout based on your hardware
4. **Check Results**: Always review the optimization reports
5. **Validate**: Test optimized parameters on a separate validation set

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or number of trials
2. **Slow Optimization**: Use fewer trials or shorter timeout
3. **Poor Results**: Check parameter ranges and increase trials
4. **Missing Results**: Ensure optimization completed successfully

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run optimization with debug logging
optimizer = HyperparameterOptimizer()
optimizer.load_data()
study = optimizer.optimize_multi_label()
```

## Integration with Training Scripts

The optimized parameters can be directly integrated into the existing training scripts:

### Multi-label Training
```python
# In train_multi_label_model.py
from utils.apply_optimized_params import load_optimized_params, apply_multi_label_params

# Load optimized parameters
params = load_optimized_params("multi_label_optimization_20241201_143022")
training_params = apply_multi_label_params(params)

# Use in training
trainer = NeuralNetworkTrainer()
trainer.load_data()
trainer.create_model(**training_params)
trainer.train_model(**training_params)
```

### Per-book Training
```python
# In train_model_per_book.py
from utils.apply_optimized_params import load_optimized_params, apply_per_book_params

# Load optimized parameters for specific book
params = load_optimized_params("book_1_optimization_20241201_143022")
training_params = apply_per_book_params(params)

# Use in training
trainer = IndividualBookTrainer()
trainer.load_data()
trainer.train_book_model("book_1", **training_params)
```

## Performance Comparison

After optimization, you can compare the performance of optimized vs default parameters:

1. **Default Parameters**: Run training with default hyperparameters
2. **Optimized Parameters**: Run training with optimized hyperparameters
3. **Compare Results**: Analyze the performance improvement

Expected improvements:
- **Multi-label Model**: 2-5% improvement in F1 score
- **Per-book Models**: 3-7% improvement in individual book F1 scores
- **Training Efficiency**: Better convergence and reduced overfitting 