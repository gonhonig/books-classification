# Multi-Label Model - Per-Book Single-Label vs Multi-Label Performance Report

## Summary
- **Total test samples**: 1519
- **Single-label samples**: 747 (49.2%)
- **Multi-label samples**: 772 (50.8%)
- **Model Type**: Unified Multi-Label Neural Network
- **Training Approach**: Single model for all books with multi-label output

## Performance Overview

### Overall Model Performance Comparison

| Book | Overall Accuracy | Overall F1 | Single-Label F1 | Multi-Label F1 | Performance Pattern |
|------|------------------|------------|-----------------|----------------|-------------------|
| **Anna Karenina** | 0.881 | 0.879 | 0.393 | **0.922** | Multi-label excels |
| **Frankenstein** | 0.848 | 0.829 | **0.911** | 0.678 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.877 | 0.748 | **0.864** | 0.662 | Single-label excels |
| **Wuthering Heights** | 0.845 | 0.845 | 0.597 | **0.894** | Multi-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.881
- **Precision**: 0.899
- **Recall**: 0.860
- **F1 Score**: 0.879

#### Single-Label Performance
- **Accuracy**: 0.901
- **Precision**: 0.300
- **Recall**: 0.571
- **F1 Score**: 0.393
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.861
- **Precision**: 0.972
- **Recall**: 0.877
- **F1 Score**: 0.922
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.529
- **Multi-label performs better** by 0.529 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.848
- **Precision**: 0.837
- **Recall**: 0.821
- **F1 Score**: 0.829

#### Single-Label Performance
- **Accuracy**: 0.896
- **Precision**: 0.898
- **Recall**: 0.923
- **F1 Score**: 0.911
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.802
- **Precision**: 0.716
- **Recall**: 0.644
- **F1 Score**: 0.678
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.233
- **Single-label performs better** by 0.233 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.877
- **Precision**: 0.835
- **Recall**: 0.678
- **F1 Score**: 0.748

#### Single-Label Performance
- **Accuracy**: 0.942
- **Precision**: 0.926
- **Recall**: 0.811
- **F1 Score**: 0.864
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.813
- **Precision**: 0.762
- **Recall**: 0.585
- **F1 Score**: 0.662
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.202
- **Single-label performs better** by 0.202 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.845
- **Precision**: 0.832
- **Recall**: 0.858
- **F1 Score**: 0.845

#### Single-Label Performance
- **Accuracy**: 0.866
- **Precision**: 0.517
- **Recall**: 0.705
- **F1 Score**: 0.597
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.825
- **Precision**: 0.904
- **Recall**: 0.883
- **F1 Score**: 0.894
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.297
- **Multi-label performs better** by 0.297 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it appears alongside other books
- **Interpretation**: Wuthering Heights's distinctive gothic style and emotional intensity is more recognizable in multi-label contexts

---

## Comparative Analysis

### Model Performance Patterns

1. **Multi-Label Specialists**:
   - Books that perform significantly better in multi-label contexts
   - Have writing styles that become more distinctive when contrasted with others
   - Examples: Anna Karenina, Wuthering Heights

2. **Single-Label Specialists**:
   - Books that perform significantly better in single-label contexts
   - Have very distinctive styles that are easily recognizable in isolation
   - Examples: Frankenstein, Alice in Wonderland

### Writing Style Analysis

- **Anna Karenina & Wuthering Heights**: Complex, emotionally intense narratives with distinctive authorial voices that become more apparent when contrasted with other styles
- **Frankenstein & Alice in Wonderland**: Highly distinctive genres (gothic horror vs. children's fantasy) with unique thematic elements that are immediately recognizable

### Practical Implications

1. **For Multi-Label Classification**: Anna Karenina and Wuthering Heights models are more reliable when multiple books are present
2. **For Single-Label Classification**: Frankenstein and Alice in Wonderland models are more reliable when only one book is present
3. **Overall**: The unified multi-label model achieves strong performance (84-88% accuracy) for individual book identification

## Methodology Notes

- **Model Architecture**: Unified multi-label neural network with sigmoid outputs
- **Training Data**: Pre-existing dataset splits with aligned embeddings
- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis
- **Threshold**: 0.5 probability threshold for binary classification
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation
- **Approach**: Single model handles all books simultaneously, leveraging shared representations
