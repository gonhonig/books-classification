# Books Classification Project

## Overview
Multi-label classification of English book sentences using semantic embeddings and advanced machine learning techniques. This project implements a 6-step methodology to classify sentences from four classic books: Anna Karenina, The Adventures of Alice in Wonderland, Wuthering Heights, and Frankenstein.

## 🎯 Final Results

### Best Performing Model
- **Multi-label Classifier with KNN Features**: 99.95% accuracy
- **Contrastive Learning Orchestration**: 58.10% average accuracy
- **Semantic Embedding Baseline**: 26.1% accuracy

### Dataset
- **Balanced Dataset**: 14,750 samples (vs 21,871 imbalanced)
- **Distribution**: Much more balanced across all books
- **Split**: 70/15/15 train/val/test

## 📁 Project Structure

```
books-classification/
├── configs/
│   └── config.yaml              # Main configuration
├── data/
│   ├── processed_dataset_balanced/  # Balanced dataset
│   ├── features_knn/               # KNN features
│   ├── metadata_balanced.json      # Balanced dataset metadata
│   └── similarity_test_pairs.json  # Similarity test pairs
├── experiments/
│   ├── semantic_embedding/         # Fine-tuned semantic model
│   ├── multi_label_classifier_knn/ # Multi-label classifier results
│   ├── contrastive_orchestration/  # Contrastive learning results
│   └── evaluation_results/         # Final evaluation results
├── models/
│   ├── semantic_embedding_model.py
│   └── multi_label_classifier.py
├── utils/
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── semantic_analysis.py
│   └── visualization.py
└── [Core Pipeline Scripts]
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_full_pipeline.py
```

### 3. Individual Steps
```bash
# Step 1: Data Preparation
python data/prepare_data_balanced.py

# Step 4: Feature Extraction
python extract_features_knn.py

# Step 5: Model Training
python train_multi_label_classifier_knn.py
python train_contrastive_models.py

# Step 6: Evaluation
python evaluate_models.py
```

## 📊 Methodology

### 6-Step Approach

1. **Data Preparation** ✅
   - Clean and preprocess English book descriptions
   - Balanced sampling (5,000 sentences per book)
   - 14,750 total samples with balanced distribution

2. **Semantic Embedding Model Selection** ✅
   - Tested 4 candidate models
   - Selected: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - 98.75% accuracy on similarity test pairs

3. **Fine-tune Selected Model** ✅
   - Contrastive learning with triplet loss
   - 10 epochs, learning rate 2e-5
   - Improved semantic understanding

4. **Feature Extraction (KNN)** ✅
   - KNN-based similarity computation
   - 4 similarity scores + 4 multi-label belonging + metadata
   - 70.07% KNN accuracy with balanced data

5. **Model Training** ✅
   - **Multi-label Classifier**: Random Forest with KNN features
   - **Contrastive Learning**: 4 separate models with triplet loss
   - Both approaches trained on balanced dataset

6. **Evaluation & Comparison** ✅
   - Comprehensive comparison of all approaches
   - Visualization and performance analysis
   - Final model selection

## 🏆 Results Summary

### Model Performance (Balanced Dataset)
| Model | Accuracy | Hamming Loss | Notes |
|-------|----------|--------------|-------|
| Multi-label Classifier (KNN) | 99.95% | 0.0001 | **Winner** |
| Contrastive Learning | 58.10% | - | Runner-up |
| Semantic Embedding | 26.1% | - | Baseline |

### Key Findings
- **Balanced dataset** provides realistic evaluation
- **KNN features** are highly effective for multi-label classification
- **Multi-label classifier** achieves near-perfect accuracy
- **Contrastive learning** shows good generalization

## 📈 Technical Highlights

### Balanced Dataset Impact
- **KNN Accuracy**: 84.44% → 70.07% (more realistic)
- **Multi-label Performance**: Maintained 99.95% accuracy
- **Contrastive Learning**: 82.88% → 58.10% (more realistic)

### Feature Engineering
- **KNN Similarity Scores**: 4 features per book
- **Multi-label Belonging**: 4 binary features
- **KNN Metadata**: Confidence, best book, etc.
- **Total Features**: 10 numeric features

### Model Architecture
- **Multi-label Classifier**: Random Forest with 100 trees
- **Contrastive Learning**: 4 separate models with triplet loss
- **Semantic Embedding**: Fine-tuned transformer model

## 📋 Files Overview

### Core Pipeline Scripts
- `run_full_pipeline.py` - Complete pipeline execution
- `extract_features_knn.py` - KNN feature extraction
- `train_multi_label_classifier_knn.py` - Multi-label classifier training
- `train_contrastive_models.py` - Contrastive learning training
- `evaluate_models.py` - Model evaluation and comparison

### Data & Configuration
- `data/processed_dataset_balanced/` - Balanced dataset
- `configs/config.yaml` - Main configuration
- `data/features_knn/` - Extracted KNN features

### Results & Documentation
- `BALANCED_DATASET_RESULTS.md` - Detailed results
- `QUICK_REFERENCE.md` - Quick reference guide
- `experiments/` - All model outputs and results

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 📝 Usage Examples

### Train Multi-label Classifier
```bash
python train_multi_label_classifier_knn.py --config configs/config.yaml
```

### Train Contrastive Models
```bash
python train_contrastive_models.py --config configs/config.yaml
```

### Evaluate All Models
```bash
python evaluate_models.py --config configs/config.yaml
```

## 🤝 Contributing

1. Follow the 6-step methodology
2. Use balanced dataset for fair evaluation
3. Document all experiments and results
4. Maintain clean project structure

## 📄 License

This project is for educational and research purposes.

---

**Last Updated**: [Current Date]
**Status**: ✅ Complete with balanced dataset evaluation 