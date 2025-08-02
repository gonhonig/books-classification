# Project Methodology: Multi-label Classification of English Book Sentences

## Overview
This project implements a comprehensive approach to classify English book sentences using semantic embeddings and multi-label classification. The methodology follows a 6-step process that combines semantic understanding with traditional machine learning.

## 6-Step Methodology

### Step 1: Data Preparation ✅ COMPLETED
**Objective**: Clean and preprocess English book descriptions for training
- **Input**: Raw book corpus (4 selected books)
- **Process**: 
  - Text cleaning and normalization
  - Sentence segmentation and filtering
  - Train/validation/test split (70/15/15)
  - Metadata creation and storage
- **Output**: Processed dataset with 21,871 training samples
- **Files**: `data/dataset/`, `data/metadata.json`

### Step 1.5: Semantic Augmented Dataset Creation ✅ COMPLETED
**Objective**: Create multi-label dataset using semantic pairs
- **Input**: Deduplicated corpus + semantic pairs
- **Process**:
  - Start with all sentences from deduplicated corpus (31,760 sentences)
  - Mark each sentence with its original book (1 label)
  - Use semantic pairs to add cross-book similarities
  - Create multi-label annotations (1-4 labels per sentence)
- **Output**: Semantic augmented dataset with multi-label format
- **Dataset Statistics**:
  - Total sentences: 31,760
  - Train/Val/Test: 22,231/4,764/4,765
  - Label distribution: 1-label (83.8%), 2-labels (11.0%), 3-labels (4.0%), 4-labels (1.3%)
- **Files**: `data/dataset/` (multi-label format), `data/semantic_augmented/`

### Step 2: Semantic Embedding Model Selection ✅ COMPLETED
**Objective**: Test and select the best semantic embedding model
- **Approach**: Evaluate 4 candidate models using similarity test pairs
- **Models Tested**:
  - `sentence-transformers/all-MiniLM-L6-v2` (91.25% accuracy)
  - `sentence-transformers/all-mpnet-base-v2` (92.50% accuracy)
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (98.75% accuracy) ⭐ **SELECTED**
  - `sentence-transformers/paraphrase-MiniLM-L3-v2` (95.00% accuracy)
- **Test Data**: 80 similarity test pairs (40 similar, 40 dissimilar)
- **Selection Criteria**: Accuracy, score separation, model size, inference time
- **Files**: `experiments/model_selection/semantic_model_comparison.json`

### Step 3: Fine-tune Selected Model ✅ COMPLETED
**Objective**: Fine-tune the selected semantic embedding model
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Training Data**: 40 similar sentence pairs from similarity test pairs
- **Method**: Contrastive learning with triplet loss
- **Parameters**: 10 epochs, learning rate 2e-5, temperature 0.1, margin 0.3
- **Output**: Fine-tuned model with improved semantic understanding
- **Files**: `experiments/semantic_embedding/semantic_embedding_model.pt`

### Step 4: Feature Extraction & Dataset Construction ✅ COMPLETED
**Objective**: Extract features using KNN approach and create augmented dataset
- **Approach**: KNN-based similarity computation with embedding caching
- **Method**:
  - Extract embeddings for all sentences using fine-tuned model
  - Cache embeddings for future use (`data/embeddings_cache.npz`)
  - Compute KNN similarities (k=5 neighbors)
  - Apply multi-label belonging logic:
    - Best score always gets 1
    - All scores within 0.2 of best get 1
    - Original true book always gets 1
- **Features Created**:
  - 4 similarity scores (one per book)
  - 4 multi-label belonging indicators
  - KNN metadata (best book, confidence, etc.)
- **Results**:
  - KNN Accuracy: 84.44%
  - Mean books per sentence: 1.24
  - Multi-book ratio: 21.7%
- **Files**: `data/features_knn/augmented_dataset.csv`

**Note**: This step has been superseded by the semantic augmented dataset approach, which provides better multi-label classification capabilities.

### Step 5: Train Competing Models ⏳ IN PROGRESS
**Objective**: Train and compare different classification approaches
- **Approach 1**: Multi-label Classifier
  - Single model for all categories
  - Features: Similarity scores + multi-label belonging
  - Algorithms: Random Forest, Logistic Regression, SVM
- **Approach 2**: Contrastive Learning Orchestration
  - 4 separate models (one per book)
  - Triplet loss training
  - Ensemble prediction
- **Evaluation Metrics**: Precision, Recall, F1-Score, Hamming Loss

### Step 6: Compare, Visualize, and Conclude ⏳ PENDING
**Objective**: Comprehensive comparison and final model selection
- **Comparison Dimensions**:
  - Overall accuracy and per-category performance
  - Computational efficiency (training/inference time)
  - Model interpretability
  - Multi-label classification quality
- **Visualization**: Confusion matrices, performance charts, feature importance
- **Output**: Final model selection and deployment recommendations

## Key Innovations

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

## Technical Architecture

### Data Flow
1. Raw Books → Preprocessing → Processed Dataset
2. Processed Dataset → Similarity Pairs → Model Selection
3. Selected Model → Fine-tuning → Enhanced Embeddings
4. Enhanced Embeddings → KNN Feature Extraction → Augmented Dataset
5. Augmented Dataset → Multi-label Classification → Final Models
6. Final Models → Comparison → Best Model Selection

### Key Technologies
- **Semantic Embeddings**: Sentence Transformers
- **Fine-tuning**: Contrastive Learning with Triplet Loss
- **Feature Extraction**: KNN with Cosine Similarity
- **Classification**: Multi-label ML algorithms
- **Caching**: NumPy compressed format
- **Evaluation**: Comprehensive metrics suite

## Current Status
- **Completed Steps**: 1-4 (Data Prep, Model Selection, Fine-tuning, Feature Extraction)
- **Current Focus**: Step 5 (Training Competing Models)
- **Next Milestone**: Multi-label classifier training and evaluation

## Files Structure
```
books-classification/
├── data/
│   ├── processed_dataset/          # Step 1 output
│   ├── embeddings_cache.npz        # Step 4 caching
│   ├── features_knn/      # Step 4 output
│   └── similarity_test_pairs.json  # Step 2 data
├── experiments/
│   ├── model_selection/            # Step 2 results
│   └── semantic_embedding/         # Step 3 output
├── models/                         # Model implementations
├── utils/                          # Utility functions
└── configs/                        # Configuration files
```

## Success Metrics
- **Semantic Model Accuracy**: 98.75% (exceeded target)
- **KNN Feature Accuracy**: 84.44% (significant improvement)
- **Multi-label Realism**: 21.7% multi-book ratio (realistic)
- **Performance**: Embedding caching reduces time by ~90%
- **Quality**: Handles edge cases and uniform distributions 