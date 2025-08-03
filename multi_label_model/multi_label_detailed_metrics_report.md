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
| **Anna Karenina** | 0.874 | 0.872 | 0.412 | **0.916** | Multi-label excels |
| **Frankenstein** | 0.870 | 0.855 | **0.922** | 0.737 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.903 | 0.816 | **0.883** | 0.771 | Single-label excels |
| **Wuthering Heights** | 0.853 | 0.854 | 0.617 | **0.901** | Multi-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.874
- **Precision**: 0.887
- **Recall**: 0.857
- **F1 Score**: 0.872

#### Single-Label Performance
- **Accuracy**: 0.897
- **Precision**: 0.303
- **Recall**: 0.643
- **F1 Score**: 0.412
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.851
- **Precision**: 0.968
- **Recall**: 0.870
- **F1 Score**: 0.916
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.504
- **Multi-label performs better** by 0.504 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.870
- **Precision**: 0.854
- **Recall**: 0.856
- **F1 Score**: 0.855

#### Single-Label Performance
- **Accuracy**: 0.909
- **Precision**: 0.913
- **Recall**: 0.930
- **F1 Score**: 0.922
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.832
- **Precision**: 0.746
- **Recall**: 0.728
- **F1 Score**: 0.737
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.185
- **Single-label performs better** by 0.185 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.903
- **Precision**: 0.831
- **Recall**: 0.802
- **F1 Score**: 0.816

#### Single-Label Performance
- **Accuracy**: 0.949
- **Precision**: 0.917
- **Recall**: 0.852
- **F1 Score**: 0.883
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.858
- **Precision**: 0.774
- **Recall**: 0.768
- **F1 Score**: 0.771
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.113
- **Single-label performs better** by 0.113 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.853
- **Precision**: 0.838
- **Recall**: 0.870
- **F1 Score**: 0.854

#### Single-Label Performance
- **Accuracy**: 0.869
- **Precision**: 0.523
- **Recall**: 0.752
- **F1 Score**: 0.617
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.838
- **Precision**: 0.914
- **Recall**: 0.889
- **F1 Score**: 0.901
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.284
- **Multi-label performs better** by 0.284 F1 points
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
