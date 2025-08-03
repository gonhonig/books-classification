# Books Classification Project

## 📚 Overview

This project implements a comprehensive book classification system using advanced neural networks and semantic embeddings. The goal is to classify sentences from different books using two competing approaches: **per-book binary classifiers** and **unified multi-label classification**.

## 🎯 Project Goals

- **Self-Supervised Learning**: Use unsupervised semantic similarity without manual data tagging
- **Dual Approach Comparison**: Compare per-book vs multi-label classification strategies
- **Semantic Understanding**: Use advanced embedding models to capture semantic similarities
- **Multi-label Classification**: Handle sentences that may belong to multiple books simultaneously
- **Performance Optimization**: Achieve high accuracy through hyperparameter optimization
- **Comprehensive Evaluation**: Provide detailed analysis of both approaches

## 🏗️ Project Architecture

### Two Main Approaches

#### 1. Per-Book Approach ✅ IMPLEMENTED
**Objective**: Train individual binary classifiers for each book using self-supervised learning

**Process**:
- Separate neural network for each book (Anna Karenina, Wuthering Heights, Frankenstein, Alice in Wonderland)
- Each model trained to predict whether a sentence belongs to its specific book
- No manual labeling required - uses semantic similarity for supervision
- Independent optimization and evaluation for each book

**Key Files**: `model_per_book/`, `train_model_per_book.py`

#### 2. Multi-Label Approach ✅ IMPLEMENTED
**Objective**: Train a single unified model for all books simultaneously using self-supervised learning

**Process**:
- Single neural network with 4 output heads (one per book)
- Multi-label classification where sentences can belong to multiple books
- Shared representations across all books
- No manual data tagging - uses semantic similarity for supervision

**Key Files**: `multi_label_model/`, `train_multi_label_model.py`

## 📊 Data Preparation Pipeline

### Stage 1: Sentence Gathering
**Objective**: Collect and preprocess sentences from all books

**Process**:
- Extract sentences from 4 classic books
- Clean and normalize text (remove special characters, standardize formatting)
- Remove duplicates and very short sentences
- Create balanced representation across all books

**Output**: 31,760 unique sentences from deduplicated corpus

### Stage 2: Embedding Generation
**Objective**: Create semantic embeddings using pre-trained models

**Process**:
- Use sentence-transformers library with pre-trained models
- Generate 384-dimensional embeddings for all sentences
- Cache embeddings for efficient reuse (`data/embeddings_cache_*.npz`)
- Ensure consistent embedding space across all books

**Why This Stage**: Semantic embeddings capture meaning better than bag-of-words approaches

### Stage 3: Similarity Pair Extraction
**Objective**: Identify semantically similar sentences across books

**Process**:
- Use cosine similarity to find cross-book sentence pairs
- Apply similarity threshold (0.7) to ensure quality pairs
- Generate 11,861 semantic pairs for dataset augmentation
- Store pairs in `data/semantic_pairs.json`

**Why This Stage**: Enables multi-label classification by identifying cross-book semantic relationships

### Stage 4: Balanced Augmented Dataset Creation
**Objective**: Create multi-label dataset with realistic cross-book relationships and balanced sampling

**Process**:
- Start with all sentences from deduplicated corpus
- Mark each sentence with its original book (1 label)
- Use semantic pairs to add cross-book similarities
- Create multi-label annotations (1-4 labels per sentence)
- **Balanced Sampling**: Sample similar amounts from each book while prioritizing multi-label sentences
- **Multi-label Priority**: Prefer sentences with multiple labels to ensure good performance on multi-label task
- Apply train/validation/test split (70/15/15)

**Output**: 
- 10,120 sentences with multi-label format (balanced dataset)
- Train/Validation/Test: 7,083 / 1,518 / 1,519
- Label distribution: 1-label (49.1%), 2-labels (34.4%), 3-labels (12.5%), 4-labels (4.1%)
- **Balanced representation** across all books with emphasis on multi-label examples

**Why This Stage**: Provides realistic multi-label training data that reflects actual semantic relationships between books while ensuring balanced representation and strong multi-label performance

## 🎯 Hyperparameter Optimization

### Comprehensive Optimization System ✅ IMPLEMENTED
**Objective**: Optimize all model components for best performance

**Components Optimized**:
- **Neural Network Architecture**: Number of layers, hidden dimensions
- **Training Parameters**: Learning rate, batch size, epochs, patience
- **Regularization**: Dropout rate, weight decay
- **Both Approaches**: Per-book and multi-label models optimized separately

**Method**: Bayesian optimization with Optuna

**Key Features**:
- **Multi-label Model Optimization**: Optimizes unified model parameters
- **Per-book Model Optimization**: Optimizes individual binary classifiers
- **Visualization**: Optimization history and parameter importance plots
- **Detailed Reporting**: Markdown reports with trial history and best parameters
- **Easy Integration**: Simple API to apply optimized parameters

**Key Files**: `utils/hyperparameter_optimizer.py`, `HYPERPARAMETER_OPTIMIZATION_README.md`

## 📈 Results Summary

### Performance Comparison

| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |
|------|-------------|----------------|-----------|--------|
| Anna Karenina | 0.879 | 0.880 | +0.001 | Multi-Label |
| Wuthering Heights | 0.848 | 0.852 | +0.004 | Multi-Label |
| Frankenstein | 0.847 | 0.844 | -0.003 | Per-Book |
| The Adventures of Alice in Wonderland | 0.801 | 0.797 | -0.005 | Per-Book |

### Key Insights

#### Overall Performance
- **Average Performance**: Per-Book (0.844) vs Multi-Label (0.843) - virtually identical
- **Balanced Results**: Each approach wins in 2/4 books
- **Consistent Quality**: Both approaches achieve 80%+ accuracy across all books

#### Book-Specific Analysis
- **Anna Karenina & Wuthering Heights**: Multi-label approach performs better
- **Frankenstein & Alice in Wonderland**: Per-book approach performs better
- **Interpretation**: Books with more semantic overlap benefit from shared representations

#### Multi-label vs Single-label Performance
- **Anna Karenina**: Multi-label F1 91.2% vs Single-label F1 41.3% (+49.9%)
- **Wuthering Heights**: Multi-label F1 89.7% vs Single-label F1 58.2% (+31.4%)
- **Frankenstein**: Multi-label F1 69.9% vs Single-label F1 91.7% (-21.8%)
- **Alice in Wonderland**: Multi-label F1 75.1% vs Single-label F1 86.2% (-11.1%)

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Hyperparameter Optimization
```bash
# Optimize multi-label model
python utils/run_hyperparameter_optimization.py --model_type multi_label --n_trials 30

# Optimize per-book models
python utils/run_hyperparameter_optimization.py --model_type per_book --n_trials 30

# Optimize both approaches
python utils/run_hyperparameter_optimization.py --model_type both --n_trials 30
```

### 3. Train Models
```bash
# Train multi-label model
python multi_label_model/train_multi_label_model.py

# Train per-book models
python model_per_book/train_model_per_book.py
```

### 4. Compare Approaches
```bash
# Generate comprehensive comparison
python compare_approaches.py
```

## 📁 Project Structure

```
books-classification/
├── configs/
│   ├── config.yaml                    # Main configuration
│   └── optimized_params_config.yaml   # Optimized parameters
├── data/
│   ├── dataset/                       # Multi-label dataset splits
│   ├── semantic_augmented/            # Dataset analysis and documentation
│   ├── embeddings_cache_*.npz         # Cached embeddings
│   ├── semantic_pairs.json            # Semantic similarity pairs
│   └── corpus_deduplicated.json       # Deduplicated corpus
├── model_per_book/                    # Per-book approach
│   ├── train_model_per_book.py        # Training script
│   ├── *_best_model.pth              # Trained models for each book
│   ├── results.json                   # Training results
│   ├── model_comparison.png           # Performance visualization
│   └── README.md                      # Detailed documentation
├── multi_label_model/                 # Multi-label approach
│   ├── train_multi_label_model.py     # Training script
│   ├── best_neural_network.pth        # Trained unified model
│   ├── neural_network_results.json    # Training results
│   └── README.md                      # Detailed documentation
├── utils/
│   ├── hyperparameter_optimizer.py    # Optimization engine
│   ├── run_hyperparameter_optimization.py  # Optimization runner
│   ├── apply_optimized_params.py      # Parameter application
│   ├── evaluation.py                  # Evaluation utilities
│   └── visualization.py               # Plotting utilities
├── compare_approaches.py              # Main comparison script
├── approach_comparison_report.md      # Comparison results
├── HYPERPARAMETER_OPTIMIZATION_README.md  # Optimization documentation
├── SEMANTIC_AUGMENTED_DATASET_README.md  # Dataset documentation
└── requirements.txt                   # Dependencies
```

## 🔧 Key Technologies

### Neural Networks
- **Framework**: PyTorch
- **Architecture**: Multi-layer perceptrons with ReLU activation
- **Regularization**: BatchNorm, Dropout, Weight Decay
- **Optimization**: Adam optimizer with learning rate scheduling

### Semantic Embeddings
- **Framework**: Sentence Transformers
- **Model**: Pre-trained multilingual models
- **Dimension**: 384-dimensional embeddings
- **Caching**: NumPy compressed format for efficiency

### Hyperparameter Optimization
- **Framework**: Optuna
- **Method**: Bayesian optimization
- **Parameters**: Architecture, training, regularization
- **Visualization**: Optimization history and importance plots

### Evaluation Metrics
- **Overall**: Accuracy, Precision, Recall, F1-Score
- **Multi-label**: Hamming Loss, Exact Match Ratio
- **Per-book**: Individual binary classification metrics

## 📊 Dataset Information

### Books Included
1. **Anna Karenina** - Leo Tolstoy
2. **Frankenstein** - Mary Shelley
3. **The Adventures of Alice in Wonderland** - Lewis Carroll
4. **Wuthering Heights** - Emily Brontë

### Dataset Statistics
- **Total Sentences**: 10,120 (balanced semantic augmented dataset)
- **Train/Validation/Test Split**: 7,083 / 1,518 / 1,519
- **Multi-label Distribution**:
  - 1 label: 4,966 sentences (49.1%)
  - 2 labels: 3,479 sentences (34.4%)
  - 3 labels: 1,264 sentences (12.5%)
  - 4 labels: 411 sentences (4.1%)

### Cross-Book Similarity Analysis
- **Anna Karenina ↔ Wuthering Heights**: 14.0% similarity (strongest)
- **Frankenstein ↔ Alice in Wonderland**: 6.6% similarity (weakest)
- **Most Universal**: Expressions like "Oh dear!", "Well!", "she said aloud"

### Book Distribution (Balanced Dataset)
- **Anna Karenina**: 5,000 positive samples (49.4%)
- **Wuthering Heights**: 5,000 positive samples (49.4%)
- **Frankenstein**: 4,443 positive samples (43.9%)
- **The Adventures of Alice in Wonderland**: 2,917 positive samples (28.8%)

## 🎯 Key Innovations

### 1. Self-Supervised Learning Approach
- **No Manual Labeling**: Uses semantic similarity for supervision instead of manual data tagging
- **Per-Book Models**: Specialized binary classifiers for each book
- **Multi-Label Model**: Unified model with shared representations
- **Balanced Evaluation**: Comprehensive comparison across multiple metrics

### 2. Semantic Augmented Dataset
- **Cross-Book Relationships**: Based on actual semantic similarity
- **Multi-label Format**: Realistic representation of book relationships
- **Complete Coverage**: All sentences from deduplicated corpus included
- **Balanced Sampling**: Similar representation from each book with multi-label priority

### 3. Advanced Hyperparameter Optimization
- **Bayesian Optimization**: Efficient parameter search with Optuna
- **Comprehensive Coverage**: Architecture, training, and regularization parameters
- **Visualization**: Optimization history and parameter importance analysis

### 4. Detailed Performance Analysis
- **Overall Metrics**: Standard classification metrics
- **Multi-label Analysis**: Performance on sentences belonging to multiple books
- **Single-label Analysis**: Performance on sentences belonging to single books
- **Book-Specific Insights**: Detailed breakdown for each book

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Optuna (for hyperparameter optimization)
- Sentence-Transformers

## 📝 Usage Examples

### Train with Optimized Parameters
```bash
# Apply optimized parameters and train
python utils/apply_optimized_params.py --study_name multi_label_optimization_20241201_143022
python multi_label_model/train_multi_label_model.py
```

### Run Complete Comparison
```bash
# Train both approaches and compare
python model_per_book/train_model_per_book.py
python multi_label_model/train_multi_label_model.py
python compare_approaches.py
```

### Quick Performance Check
```bash
# Generate comparison report
python compare_approaches.py --quick
```

## 🤝 Contributing

1. Follow the dual approach methodology
2. Use semantic augmented dataset for fair evaluation
3. Run hyperparameter optimization for new components
4. Document all experiments and results
5. Maintain clean project structure

## 📄 License

This project is for educational and research purposes.

---

**Status**: ✅ Complete with comprehensive comparison and optimization
**Best Approach**: Balanced - Per-book for specialized performance, Multi-label for shared representations 