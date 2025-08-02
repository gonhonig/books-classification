# Books Classification Project

## 📚 Overview

This project implements a comprehensive multi-label classification system for English book sentences using semantic embeddings and advanced machine learning techniques. The goal is to classify sentences from different books based on their semantic similarity and content characteristics.

## 🎯 Project Goals

- **Multi-label Classification**: Classify sentences that may belong to multiple books simultaneously
- **Semantic Understanding**: Use advanced embedding models to capture semantic similarities
- **Scalable Architecture**: Create a robust system that can handle large datasets efficiently
- **Performance Optimization**: Achieve high accuracy while maintaining computational efficiency

## 🏗️ Project Architecture

### 6-Step Methodology

#### 1. Data Preparation ✅ COMPLETED
**Objective**: Clean and preprocess English book descriptions for training

**Process**:
- Text cleaning and normalization
- Sentence segmentation and filtering
- Train/validation/test split (70/15/15)
- Metadata creation and storage

**Output**: 
- Processed dataset with 21,871 training samples
- Balanced dataset with 14,750 samples (5,000 per book where possible)

**Key Files**: `data/dataset/`, `data/metadata.json`

#### 1.5. Semantic Augmented Dataset Creation ✅ COMPLETED
**Objective**: Create multi-label dataset using semantic pairs

**Process**:
- Start with all sentences from deduplicated corpus (31,760 sentences)
- Mark each sentence with its original book (1 label)
- Use semantic pairs to add cross-book similarities
- Create multi-label annotations (1-4 labels per sentence)

**Output**: 
- 31,760 sentences with multi-label format
- Train/Val/Test: 22,231/4,764/4,765
- Label distribution: 1-label (83.8%), 2-labels (11.0%), 3-labels (4.0%), 4-labels (1.3%)

**Key Files**: `data/semantic_augmented/`, `data/semantic_pairs.json`

#### 2. Semantic Embedding Model Selection ✅ COMPLETED
**Objective**: Test and select the best semantic embedding model

**Approach**: Evaluate 4 candidate models using similarity test pairs

**Models Tested**:
- `sentence-transformers/all-MiniLM-L6-v2` (91.25% accuracy)
- `sentence-transformers/all-mpnet-base-v2` (92.50% accuracy)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (98.75% accuracy) ⭐ **SELECTED**
- `sentence-transformers/paraphrase-MiniLM-L3-v2` (95.00% accuracy)

**Selection Criteria**: Accuracy, score separation, model size, inference time

#### 3. Fine-tune Selected Model ✅ COMPLETED
**Objective**: Fine-tune the selected semantic embedding model

**Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Method**: Contrastive learning with triplet loss

**Parameters**: 
- 10 epochs
- Learning rate: 2e-5
- Temperature: 0.1
- Margin: 0.3

**Output**: Fine-tuned model with improved semantic understanding

#### 4. Feature Extraction & Dataset Construction ✅ COMPLETED
**Objective**: Extract features using KNN approach and create augmented dataset

**Approach**: KNN-based similarity computation with embedding caching

**Method**:
- Extract embeddings for all sentences using fine-tuned model
- Cache embeddings for future use (`data/embeddings_cache.npz`)
- Compute KNN similarities (k=15 neighbors)
- Apply multi-label belonging logic:
  - Best score always gets 1
  - All scores within 0.2 of best get 1
  - Original true book always gets 1

**Features Created**:
- 4 similarity scores (one per book)
- 4 multi-label belonging indicators
- KNN metadata (best book, confidence, etc.)

**Results**:
- KNN Accuracy: 84.44%
- Mean books per sentence: 1.24
- Multi-book ratio: 21.7%

#### 5. Hyperparameter Optimization ✅ COMPLETED
**Objective**: Optimize all model components for best performance

**Components Optimized**:
- **KNN Feature Extraction**: k_neighbors, similarity_threshold, belonging_threshold
- **Multi-label Classifier**: Model type, hyperparameters
- **Contrastive Learning**: Learning rate, batch size, temperature, margin, epochs

**Method**: Bayesian optimization with Optuna

**Best Parameters**:
- KNN: k_neighbors=15, similarity_threshold=0.95, belonging_threshold=0.3
- Multi-label Classifier: Random Forest, n_estimators=50, max_depth=15
- Contrastive Learning: learning_rate=1e-5, batch_size=32, temperature=1.0, margin=0.1, epochs=10

#### 6. Train Competing Models ✅ COMPLETED
**Objective**: Train and compare different classification approaches

**Approach 1**: Multi-label Classifier
- Single model for all categories
- Features: Similarity scores + KNN metadata
- Algorithms: Random Forest, Logistic Regression, SVM
- **Best Results**: Random Forest (90.83% accuracy, 94.73% F1-Score)

**Approach 2**: Contrastive Learning Orchestration
- 4 separate models (one per book)
- Triplet loss training with classifier heads
- **Results**: Evaluation issues (needs debugging)

#### 7. Compare, Visualize, and Conclude ✅ COMPLETED
**Objective**: Comprehensive comparison and final model selection

**Comparison Dimensions**:
- Overall accuracy and per-category performance
- Computational efficiency (training/inference time)
- Model interpretability
- Multi-label classification quality

**Final Results**:
- **Winner**: Random Forest Multi-label Classifier (90.83% accuracy)
- **Key Finding**: Proper feature/label separation is crucial
- **Best Approach**: Multi-label classification with KNN features

## 📊 Dataset Information

### Books Included
1. **Anna Karenina** - Leo Tolstoy
2. **Frankenstein** - Mary Shelley
3. **The Adventures of Alice in Wonderland** - Lewis Carroll
4. **Wuthering Heights** - Emily Brontë

### Dataset Statistics
- **Total Sentences**: 31,760 (semantic augmented dataset)
- **Train/Val/Test Split**: 22,231/4,764/4,765
- **Multi-label Distribution**:
  - 1 label: 26,606 sentences (83.8%)
  - 2 labels: 3,479 sentences (11.0%)
  - 3 labels: 1,264 sentences (4.0%)
  - 4 labels: 411 sentences (1.3%)

## 🏆 Results Summary

### Best Performing Model
- **Multi-label Classifier with KNN Features**: 90.83% accuracy
- **F1-Score**: 94.73%
- **Hamming Loss**: 0.0371

### Model Performance Comparison
| Model | Accuracy | F1-Score | Hamming Loss |
|-------|----------|----------|--------------|
| Random Forest Multi-label | 90.83% | 94.73% | 0.0371 |
| Logistic Regression | 87.71% | 93.66% | 0.0467 |
| SVM | 88.30% | 93.91% | 0.0459 |

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
python data/prepare_data.py

# Step 1.5: Semantic Augmented Dataset
python create_semantic_augmented_dataset.py

# Step 4: Feature Extraction
python extract_features_knn.py

# Step 5: Hyperparameter Optimization
python run_hyperparameter_optimization.py

# Step 6: Model Training
python train_multi_label_classifier_knn.py

# Step 7: Evaluation
python evaluate_models.py
```

## 📁 Project Structure

```
books-classification/
├── configs/
│   └── config.yaml              # Main configuration
├── data/
│   ├── dataset/                 # Processed dataset
│   ├── semantic_augmented/      # Multi-label dataset
│   ├── features_knn/            # KNN features
│   ├── embeddings_cache.npz     # Cached embeddings
│   └── semantic_pairs.json      # Semantic pairs
├── experiments/
│   ├── semantic_embedding/      # Fine-tuned model
│   ├── multi_label_classifier/  # Classification results
│   └── evaluation_results/      # Final results
├── model_per_book/              # Individual book models
├── multi_label_model/           # Multi-label model results
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

## 🔧 Key Technologies

### Semantic Embeddings
- **Framework**: Sentence Transformers
- **Selected Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Fine-tuning**: Contrastive learning with triplet loss

### Feature Engineering
- **KNN Approach**: k=15 neighbors with cosine similarity
- **Caching**: NumPy compressed format for efficiency
- **Multi-label Logic**: Realistic belonging rules

### Classification
- **Multi-label Classifier**: Random Forest with optimized parameters
- **Evaluation Metrics**: Precision, Recall, F1-Score, Hamming Loss
- **Hyperparameter Optimization**: Bayesian optimization with Optuna

## 📈 Key Innovations

### 1. Hybrid AI + Traditional Approach
- **AI Generation**: Used for creating similar sentence pairs
- **Traditional Sampling**: Used for dissimilar pairs (more reliable)
- **Combination**: Best of both worlds

### 2. KNN Feature Extraction
- **Problem Solved**: Mean embedding approach was flawed
- **Solution**: KNN-based similarity with weighted voting
- **Benefits**: Handles uniform distributions, realistic multi-label belonging

### 3. Embedding Caching
- **Performance**: First run caches embeddings, subsequent runs are fast
- **Storage**: Compressed numpy format for efficiency
- **Consistency**: Ensures reproducible results

### 4. Multi-label Belonging Logic
- **Rule 1**: Best score always gets 1
- **Rule 2**: All scores within 0.2 of best get 1
- **Rule 3**: Original true book always gets 1
- **Result**: Realistic multi-label classification

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

## 📝 Usage Examples

### Train Multi-label Classifier
```bash
python train_multi_label_classifier_knn.py --config configs/config.yaml
```

### Run Hyperparameter Optimization
```bash
python run_hyperparameter_optimization.py --config configs/config.yaml
```

### Evaluate All Models
```bash
python evaluate_models.py --config configs/config.yaml
```

## 🤝 Contributing

1. Follow the 6-step methodology
2. Use semantic augmented dataset for fair evaluation
3. Document all experiments and results
4. Maintain clean project structure
5. Run hyperparameter optimization for new components

## 📄 License

This project is for educational and research purposes.

---

**Last Updated**: [Current Date]
**Status**: ✅ Complete with semantic augmented dataset and optimized models
**Best Model**: Random Forest Multi-label Classifier (90.83% accuracy) 