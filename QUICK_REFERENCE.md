# Quick Reference Guide

## Project Status & Current Focus

### Current Phase: Following 6-Step Methodology
- **Current Step**: Step 5 🔄 IN PROGRESS - Train Competing Models
- **Goal**: Train multi-label classifier and contrastive learning models using improved KNN features
- **Next Steps**: Compare models and select the best approach

## 6-Step Project Methodology

### 1. Data Preparation ✅ COMPLETED
- Clean and preprocess English book descriptions
- Handle missing values and duplicates
- Prepare training data with proper labels
- **Output**: 21,871 training samples across 4 books

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

### 4. Feature Extraction & Dataset Construction ✅ COMPLETED (IMPROVED)
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

### 5. Train Competing Models 🔄 IN PROGRESS
- **Approach 1**: Multi-label Classifier
  - Single model for all categories
  - Features: Similarity scores + multi-label belonging
  - Algorithms: Random Forest, Logistic Regression, SVM
- **Approach 2**: Contrastive Learning Orchestration
  - 4 separate models (one per book)
  - Triplet loss training
  - Ensemble prediction
- **Evaluation Metrics**: Precision, Recall, F1-Score, Hamming Loss

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
- `extract_features_knn_improved.py` - Improved KNN feature extraction

### Configuration
- `configs/config.yaml` - Main configuration
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

### 5. Improved KNN Approach
- **Problem Solved**: Mean embedding approach was flawed
- **Solution**: KNN-based similarity with weighted voting
- **Benefits**: Handles uniform distributions, realistic multi-label belonging
- **Caching**: Embeddings cached for fast subsequent runs

## Current Workflow (Step 5)

### Multi-label Classifier Training
```bash
# Train multi-label classifier with improved features
python train_multi_label_classifier.py --config configs/config.yaml

# Train with specific algorithm
python train_multi_label_classifier.py --algorithm random_forest --config configs/config.yaml
```

### Contrastive Learning Training
```bash
# Train contrastive learning models
python train_contrastive_models.py --config configs/config.yaml
```

### Model Evaluation
```bash
# Evaluate all models
python evaluate_models.py --config configs/config.yaml

# Compare specific models
python evaluate_models.py --models multi_label contrastive --config configs/config.yaml
```

## Configuration Management

### Key Configuration Sections
```yaml
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

### Step 4: KNN Feature Evaluation
- KNN Accuracy: 84.44%
- Mean Confidence: 0.61
- Multi-label Distribution: 21.7% multi-book ratio
- **Feature Quality**: Realistic multi-label belonging

### Step 5: Classification Evaluation
- **Multi-label Classifier**: Precision, Recall, F1-Score, Hamming Loss
- **Contrastive Learning**: Category-specific accuracy, Triplet loss
- **Performance**: Training time, Inference time, Model size

### Step 6: Final Comparison
- Overall accuracy comparison
- Per-category performance
- Computational efficiency
- Model interpretability

## Common Commands

### Step 4: Feature Extraction (IMPROVED)
```bash
# Extract features using improved KNN approach
python extract_features_knn_improved.py --k-neighbors 5 --config configs/config.yaml

# Use cached embeddings (much faster)
python extract_features_knn_improved.py --k-neighbors 5 --config configs/config.yaml
```

### Step 5: Model Training
```bash
# Train multi-label classifier
python train_multi_label_classifier.py --config configs/config.yaml

# Train contrastive learning models
python train_contrastive_models.py --config configs/config.yaml
```

### Step 6: Evaluation
```bash
# Compare all models
python evaluate_models.py --config configs/config.yaml

# Create visualizations
python comprehensive_visualization.py
```

## Troubleshooting

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

### Current Session (Step 5)
1. ✅ Extract features using improved KNN approach
2. 🔄 Train multi-label classifier with improved features
3. ⏳ Train contrastive learning models
4. ⏳ Compare both approaches

### Upcoming Sessions
1. **Step 6**: Comprehensive comparison and visualization
2. **Final Analysis**: Model selection and deployment recommendations

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

### Improved KNN Quality Checklist
- [ ] Embeddings cached: `data/embeddings_cache.npz` exists
- [ ] KNN features generated: `data/features_knn_improved/` contains files
- [ ] Multi-label belonging: Realistic distribution (1.24 mean books per sentence)
- [ ] KNN accuracy: >80% (achieved 84.44%)
- [ ] Confidence metrics: Good separation (mean 0.61)

---

**Last Updated**: [Current Date]
**Reference**: See `PROJECT_METHODOLOGY.md` for detailed guidelines 