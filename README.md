# Book Sentence Classification with Constructive and Self-Supervised Learning

This project implements a system for classifying sentences to their source books using constructive learning and self-supervised learning approaches.

## Project Overview

The goal is to build a model that can classify sentences to their originating books from The Institutional Books Corpus. The project employs:

- **Constructive Learning**: Building knowledge incrementally through structured learning phases
- **Self-Supervised Learning**: Leveraging unlabeled data through pretext tasks
- **Multi-book Classification**: Classifying sentences to 4-5 different books

## Project Structure

```
books-classification/
├── data/                   # Data storage and preprocessing
├── models/                 # Model implementations
├── utils/                  # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── configs/                # Configuration files
├── experiments/            # Experiment tracking
└── requirements.txt        # Dependencies
```

## Key Features

- **Data Processing**: Automated handling of The Institutional Books Corpus
- **Constructive Learning**: Incremental model building with knowledge construction
- **Self-Supervised Tasks**: 
  - Masked Language Modeling
  - Next Sentence Prediction
  - Sentence Similarity Learning
- **Multi-Model Architecture**: Ensemble of specialized models
- **Evaluation Framework**: Comprehensive metrics and visualization

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download and preprocess data:
   ```bash
   python data/prepare_data.py
   ```

3. Train the model:
   ```bash
   python models/train.py
   ```

4. Evaluate results:
   ```bash
   python utils/evaluate.py
   ```

## Research Approach

### Constructive Learning
- **Phase 1**: Basic sentence understanding and feature extraction
- **Phase 2**: Book-specific pattern recognition
- **Phase 3**: Cross-book knowledge transfer
- **Phase 4**: Fine-tuning and optimization

### Self-Supervised Learning
- **Pretext Tasks**: 
  - Masked word prediction
  - Sentence pair classification
  - Document-level consistency
- **Downstream Task**: Book classification

## Model Architecture

- **Encoder**: Transformer-based sentence encoder
- **Constructive Components**: Incremental knowledge layers
- **Self-Supervised Heads**: Multiple pretext task heads
- **Classification Head**: Final book classification layer

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-book performance analysis
- Cross-validation results

## License

MIT License 