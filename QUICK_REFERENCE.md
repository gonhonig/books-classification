# Quick Reference Guide

## Project Status & Current Focus

### Current Phase: Following 6-Step Methodology
- **Current Step**: Step 1 🔄 IN PROGRESS - Balanced Data Preparation
- **Goal**: Create balanced dataset with equal sentences per book to address class imbalance
- **Next Steps**: Re-run Steps 2-5 with balanced data

## 6-Step Project Methodology

### 1. Data Preparation 🔄 IN PROGRESS (BALANCED VERSION)
- Clean and preprocess English book descriptions
- **NEW**: Balanced sampling - equal sentences per book
- **NEW**: 5,000 sentences per book (or all available if less)
- **Output**: 14,750 total samples (much more balanced distribution)
- **Status**: ✅ Balanced dataset created, ready for re-running pipeline

### 2. Semantic Embedding Model Selection ✅ COMPLETED
- Tested 4 candidate models:
  - Sentence-BERT
  - all-MiniLM-L6-v2
  - all-mpnet-base-v2
  - paraphrase-MiniLM-L3-v2
- **Selected Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (98.75% accuracy)
- **Similarity Test Pairs**: 80 pairs (40 similar, 40 dissimilar)
- **Selection Criteria**: Accuracy, score separation, model size, inference time

### 3. Fine-tune Selected Model ✅ COMPLETED
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Training Data**: 40 similar sentence pairs
- **Method**: Contrastive learning with triplet loss
- **Parameters**: 10 epochs, learning rate 2e-5, temperature 0.1, margin 0.3
- **Output**: Fine-tuned model with improved semantic understanding

### 4. Feature Extraction & Dataset Construction ✅ COMPLETED (OPTIMIZED)
- **Approach**: KNN-based similarity computation with embedding caching
- **Method**:
  - Extract embeddings for all sentences using fine-tuned model
  - Cache embeddings for future use (`data/embeddings_cache.npz`)
  - Compute KNN similarities (k=15 neighbors) ⭐ OPTIMIZED
  - Apply multi-label belonging logic:
    - Best score always gets 1
    - All scores within 0.2 of best get 1
    - Original true book always gets 1
- **Features Created**:
  - 4 similarity scores (one per book)
  - 4 multi-label belonging indicators
  - KNN metadata (best book, confidence, etc.)
- **Optimized Parameters**:
  - k_neighbors: 15 (was 5)
  - similarity_threshold: 0.95 (was 0.6)
  - belonging_threshold: 0.3 (new parameter)
- **Status**: ✅ Completed with optimized parameters

### 5. Hyperparameter Optimization ✅ COMPLETED
- **KNN Feature Extraction Optimization**:
  - k_neighbors: 3-15
  - similarity_threshold: 0.5-0.95
  - belonging_threshold: 0.3-0.8
  - **Best Parameters**: k_neighbors=15, similarity_threshold=0.95, belonging_threshold=0.3
  - **Best Accuracy**: 99.80%
- **Multi-label Classifier Optimization**:
  - Model types: Random Forest, Logistic Regression, SVM
  - Hyperparameters: n_estimators, max_depth, C, kernel, etc.
  - **Best Parameters**: Random Forest, n_estimators=50, max_depth=15, min_samples_split=5, min_samples_leaf=5
  - **Best Accuracy**: 99.95%
- **Contrastive Learning Optimization**:
  - learning_rate: 1e-5 to 2e-4
  - batch_size: 16, 32, 64
  - temperature: 0.1-1.0
  - margin: 0.1-2.0
  - epochs: 3-10
  - **Best Parameters**: learning_rate=1e-5, batch_size=32, temperature=1.0, margin=0.1, epochs=10
  - **Best Accuracy**: 59.42%
- **Method**: Bayesian optimization with Optuna
- **Output**: Optimized parameters for each component

### 6. Train Competing Models (with optimized parameters) ✅ COMPLETED
- **Approach 1**: Multi-label Classifier (KNN Features) - CORRECTED
  - Single model for all categories
  - **Features**: Similarity scores + KNN metadata (excluding belongs_to columns)
  - **Labels**: belongs_to columns (multi-label targets)
  - **Optimized Parameters**: Random Forest, n_estimators=50, max_depth=15, min_samples_split=5, min_samples_leaf=5
  - **Final Results**:
    - **Random Forest**: 90.83% accuracy, 94.73% F1-Score, 0.0371 Hamming Loss
    - **Logistic Regression**: 87.71% accuracy, 93.66% F1-Score, 0.0467 Hamming Loss  
    - **SVM**: 88.30% accuracy, 93.91% F1-Score, 0.0459 Hamming Loss
- **Approach 2**: Contrastive Learning Orchestration
  - 4 separate models (one per book)
  - Triplet loss training with classifier heads
  - **Optimized Parameters**: learning_rate=1e-5, batch_size=32, temperature=1.0, margin=0.1, epochs=10
  - **Final Results**: Evaluation issues (needs debugging)
- **Winner**: **Random Forest Multi-label Classifier** (90.83% accuracy)
- **Key Insight**: Proper feature/label separation is crucial for good performance

### 7. Compare, Visualize, and Conclude ✅ COMPLETED
- Comprehensive comparison of both approaches
- Visualization of results and performance metrics
- **Final Conclusions**:
  - **Winner**: Random Forest Multi-label Classifier (90.83% accuracy)
  - **Key Finding**: Proper feature/label separation is crucial
  - **Best Approach**: Multi-label classification with KNN features
  - **Performance**: 90.83% accuracy, 94.73% F1-Score, 0.0371 Hamming Loss

## Key Files & Their Purposes

### Core Pipeline Files
- `run_full_pipeline.py` - Main pipeline execution
- `model_selection_optimization.py` - Hyperparameter optimization
- `evaluate_models.py` - Model evaluation
- `create_similarity_pairs.py` - Generate similarity test pairs
- `extract_features_knn.py` - KNN feature extraction

### Configuration
- `configs/config.yaml` - Main configuration (updated with balanced sampling)
- `data/similarity_test_pairs.json` - Generated similarity test pairs
- `data/embeddings_cache.npz` - Cached embeddings for fast processing

### Models
- `models/semantic_embedding_model.py` - Semantic embedding implementation
- `models/multi_label_classifier.py` - Multi-label classification

### Utilities
- `utils/semantic_analysis.py` - Semantic similarity analysis
- `utils/evaluation.py` - Evaluation metrics
- `utils/visualization.py` - Results visualization

## Development Principles

### 1. Iterative Approach
- Start simple → Validate → Enhance → Repeat
- Document each experiment
- Save all results and configurations

### 2. Modular Design
- Separate data, models, evaluation
- Configuration-driven development
- Clear interfaces between components

### 3. Reproducible Research
- Version control everything
- Save model checkpoints
- Comprehensive logging

### 4. Hybrid AI Approach
- Use AI for creative tasks (similar sentence generation)
- Use simple sampling for reliable tasks (dissimilar pairs)
- Balance AI capabilities with reliability

### 5. KNN Approach
- **Problem Solved**: Mean embedding approach was flawed
- **Solution**: KNN-based similarity with weighted voting
- **Benefits**: Handles uniform distributions, realistic multi-label belonging
- **Caching**: Embeddings cached for fast subsequent runs

### 6. Balanced Dataset Approach ⭐ NEW
- **Problem Solved**: Severe class imbalance (Anna Karenina 64% vs Alice 5%)
- **Solution**: Equal sampling per book (5,000 sentences each)
- **Benefits**: Fair model comparison, realistic evaluation
- **Implementation**: `data/prepare_data_balanced.py`

## Current Workflow (Balanced Dataset)

### Balanced Dataset Statistics
```
The Adventures of Alice in Wonderland: 1,511 sentences (all available)
Anna Karenina: 5,000 sentences (randomly sampled)
Wuthering Heights: 5,000 sentences (randomly sampled)  
Frankenstein: 3,239 sentences (all available)
```

### Balanced Dataset Distribution
```
Train: 10,324 samples
- Wuthering Heights: 3,500
- Anna Karenina: 3,500  
- Frankenstein: 2,267
- Alice in Wonderland: 1,057

Val/Test: 2,213 samples each
- Wuthering Heights: 750
- Anna Karenina: 750
- Frankenstein: 486  
- Alice in Wonderland: 227
```

## Configuration Management

### Key Configuration Sections
```yaml
# Step 1: Balanced Data Preparation
data:
  balanced_sentences_per_book: 5000  # NEW: Equal sampling per book
  
# Step 4: Improved Feature Extraction (Optimized)
feature_extraction:
  method: "knn"
  k_neighbors: 15  # Optimized from hyperparameter optimization
  similarity_threshold: 0.95  # Optimized from hyperparameter optimization
  belonging_threshold: 0.3  # Optimized from hyperparameter optimization
  cache_embeddings: true

# Step 5: Multi-label Classification (Optimized)
models:
  multi_label_classifier:
    type: "single_model"
    algorithm: "random_forest"
    n_estimators: 50  # Optimized from hyperparameter optimization
    max_depth: 15  # Optimized from hyperparameter optimization
    min_samples_split: 5  # Optimized from hyperparameter optimization
    min_samples_leaf: 5  # Optimized from hyperparameter optimization
    features: ["similarity_scores", "multi_label_belonging"]
  
  contrastive_learning:
    type: "orchestration"
    models_per_category: 4
    loss_type: "triplet_loss"
    learning_rate: 1e-5  # Optimized from hyperparameter optimization
    batch_size: 32  # Optimized from hyperparameter optimization
    temperature: 1.0  # Optimized from hyperparameter optimization
    margin: 0.1  # Optimized from hyperparameter optimization
    epochs: 10  # Optimized from hyperparameter optimization
```

## Evaluation Metrics

### Step 1: Balanced Dataset Evaluation
- **Total Samples**: 14,750 (vs 21,871 imbalanced)
- **Class Distribution**: Much more balanced
- **Alice in Wonderland**: 1,511 samples (10.2% vs 5% before)
- **Anna Karenina**: 5,000 samples (33.9% vs 64% before)
- **Wuthering Heights**: 5,000 samples (33.9% vs 21% before)
- **Frankenstein**: 3,239 samples (22.0% vs 10% before)

### Step 4: KNN Feature Evaluation (Previous Results)
- KNN Accuracy: 84.44%
- Mean Confidence: 0.61
- Multi-label Distribution: 21.7% multi-book ratio
- **Feature Quality**: Realistic multi-label belonging

### Step 5: Classification Evaluation (Previous Results)
- **Multi-label Classifier**: 100% accuracy, 0.0000 Hamming Loss, 1.24 avg predictions/sentence
- **Contrastive Learning**: 82.88% average accuracy across 4 books
- **Winner**: KNN Multi-label Classifier (Perfect accuracy)
- **Analysis**: KNN uses pre-computed features, contrastive shows realistic generalization

## Common Commands

### Step 1: Balanced Data Preparation
```bash
# Create balanced dataset
python data/prepare_data_balanced.py --force --config configs/config.yaml

# Check balanced distribution
python -c "from datasets import load_from_disk; import pandas as pd; ds = load_from_disk('data/processed_dataset_balanced'); print(pd.Series(ds['train']['book_id']).value_counts())"
```

### Step 4: Feature Extraction (KNN) - RE-RUN NEEDED
```bash
# Extract features using KNN approach (with balanced data)
python extract_features_knn.py --k-neighbors 5 --config configs/config.yaml

# Use cached embeddings (much faster)
python extract_features_knn.py --k-neighbors 5 --config configs/config.yaml
```

### Step 5: Hyperparameter Optimization ⭐ NEW
```bash
# Complete optimization (all components)
python run_hyperparameter_optimization.py --config configs/config.yaml

# Individual component optimization
python run_hyperparameter_optimization.py --component knn --knn-trials 50
python run_hyperparameter_optimization.py --component mlc --mlc-trials 50
python run_hyperparameter_optimization.py --component contrastive --contrastive-trials 30

# Quick optimization (fewer trials)
python run_hyperparameter_optimization.py --knn-trials 20 --mlc-trials 20 --contrastive-trials 15
```

### Step 6: Model Training (with optimized parameters) - RE-RUN NEEDED
```bash
# Train multi-label classifier (with optimized parameters)
python train_multi_label_classifier_knn.py --config configs/config.yaml

# Train contrastive learning models (with optimized parameters)
python train_contrastive_models.py --config configs/config.yaml
```

### Step 7: Evaluation - RE-RUN NEEDED
```bash
# Compare all models (with optimized parameters)
python evaluate_models.py --config configs/config.yaml

# Create visualizations
python comprehensive_visualization.py
```

## Troubleshooting

### Step 1 Issues
- **Balanced Dataset**: Ensure equal sampling per book
- **Data Quality**: Check sentence preprocessing
- **Distribution**: Verify balanced class distribution

### Step 4 Issues
- **Memory Issues**: Reduce batch size, use smaller models
- **Caching Issues**: Delete `data/embeddings_cache.npz` to regenerate
- **Performance Issues**: Use GPU acceleration, optimize batch processing
- **KNN Issues**: Adjust k_neighbors parameter, check similarity computation

### Step 5 Issues
- **Feature Issues**: Ensure improved KNN features are available
- **Training Issues**: Check data format, verify feature columns
- **Memory Issues**: Reduce batch size, use smaller models

## Next Steps

### Current Session (Balanced Dataset + Hyperparameter Optimization)
1. ✅ Create balanced dataset (14,750 samples, much more balanced)
2. ✅ Re-run Step 4: Feature extraction with balanced data
3. ✅ Re-run Step 5: Model training with balanced data
4. ⏳ **NEW**: Step 5.5: Hyperparameter optimization for all components
5. ⏳ Re-run Step 6: Model training with optimized parameters
6. ⏳ Re-run Step 7: Evaluation and comparison

### Upcoming Sessions
1. **Hyperparameter Optimization**: Optimize KNN, multi-label classifier, and contrastive learning
2. **Re-run Pipeline**: Steps 6-7 with optimized parameters
3. **Compare Results**: Optimized vs non-optimized performance
4. **Final Analysis**: Model selection and deployment recommendations

## Communication Guidelines

### When Starting New Work
1. Reference this quick guide
2. Check current project status
3. Follow methodology principles
4. Document all changes

### When Reporting Progress
1. Use consistent metrics
2. Compare against baselines
3. Include configuration details
4. Save all results

### Balanced Dataset Checklist
- [ ] Balanced dataset created: `data/processed_dataset_balanced` exists
- [ ] Equal sampling: ~5,000 sentences per book (or all available)
- [ ] Class distribution: Much more balanced than before
- [ ] Data quality: Preprocessed sentences are clean
- [ ] Split ratios: 70/15/15 train/val/test maintained

---

**Last Updated**: [Current Date]
**Reference**: See `PROJECT_METHODOLOGY.md` for detailed guidelines 