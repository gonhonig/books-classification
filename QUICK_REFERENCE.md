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

### 4. Feature Extraction & Dataset Construction ⏳ PENDING (RE-RUN NEEDED)
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
- **Status**: Needs re-run with balanced dataset

### 5. Train Competing Models ⏳ PENDING (RE-RUN NEEDED)
- **Approach 1**: Multi-label Classifier (KNN Features)
  - Single model for all categories
  - Features: Similarity scores + multi-label belonging + KNN metadata
  - **Previous Results**: 100% accuracy, 0.0000 Hamming Loss (with imbalanced data)
- **Approach 2**: Contrastive Learning Orchestration
  - 4 separate models (one per book)
  - Triplet loss training with classifier heads
  - **Previous Results**: 82.88% average accuracy (with imbalanced data)
- **Status**: Needs re-run with balanced dataset

### 6. Compare, Visualize, and Conclude ⏳ PENDING
- Comprehensive comparison of both approaches
- Visualization of results and performance metrics
- Final conclusions and model selection

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
  
# Step 4: Improved Feature Extraction
feature_extraction:
  method: "knn"
  k_neighbors: 5
  similarity_threshold: 0.6
  cache_embeddings: true

# Step 5: Multi-label Classification
models:
  multi_label_classifier:
    type: "single_model"
    algorithm: "random_forest"
    features: ["similarity_scores", "multi_label_belonging"]
  
  contrastive_learning:
    type: "orchestration"
    models_per_category: 4
    loss_type: "triplet_loss"
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

### Step 5: Model Training - RE-RUN NEEDED
```bash
# Train multi-label classifier (with balanced data)
python train_multi_label_classifier.py --config configs/config.yaml

# Train contrastive learning models (with balanced data)
python train_contrastive_models.py --config configs/config.yaml
```

### Step 6: Evaluation - RE-RUN NEEDED
```bash
# Compare all models (with balanced data)
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

### Current Session (Balanced Dataset)
1. ✅ Create balanced dataset (14,750 samples, much more balanced)
2. 🔄 Re-run Step 4: Feature extraction with balanced data
3. ⏳ Re-run Step 5: Model training with balanced data
4. ⏳ Re-run Step 6: Evaluation and comparison

### Upcoming Sessions
1. **Re-run Pipeline**: Steps 4-6 with balanced dataset
2. **Compare Results**: Balanced vs imbalanced performance
3. **Final Analysis**: Model selection and deployment recommendations

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