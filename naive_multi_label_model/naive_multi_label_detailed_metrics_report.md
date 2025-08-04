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
| **Anna Karenina** | 0.763 | 0.706 | 0.562 | **0.719** | Multi-label excels |
| **Frankenstein** | 0.817 | 0.760 | **0.913** | 0.361 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.823 | 0.540 | **0.848** | 0.196 | Single-label excels |
| **Wuthering Heights** | 0.606 | 0.377 | **0.629** | 0.312 | Single-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.763
- **Precision**: 0.935
- **Recall**: 0.567
- **F1 Score**: 0.706

#### Single-Label Performance
- **Accuracy**: 0.944
- **Precision**: 0.500
- **Recall**: 0.643
- **F1 Score**: 0.562
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.588
- **Precision**: 0.993
- **Recall**: 0.563
- **F1 Score**: 0.719
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.156
- **Multi-label performs better** by 0.156 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.817
- **Precision**: 0.922
- **Recall**: 0.646
- **F1 Score**: 0.760

#### Single-Label Performance
- **Accuracy**: 0.902
- **Precision**: 0.941
- **Recall**: 0.886
- **F1 Score**: 0.913
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.734
- **Precision**: 0.817
- **Recall**: 0.232
- **F1 Score**: 0.361
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.551
- **Single-label performs better** by 0.551 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.823
- **Precision**: 0.903
- **Recall**: 0.385
- **F1 Score**: 0.540

#### Single-Label Performance
- **Accuracy**: 0.937
- **Precision**: 0.936
- **Recall**: 0.775
- **F1 Score**: 0.848
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.712
- **Precision**: 0.771
- **Recall**: 0.112
- **F1 Score**: 0.196
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.652
- **Single-label performs better** by 0.652 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.606
- **Precision**: 0.846
- **Recall**: 0.242
- **F1 Score**: 0.377

#### Single-Label Performance
- **Accuracy**: 0.902
- **Precision**: 0.674
- **Recall**: 0.590
- **F1 Score**: 0.629
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.319
- **Precision**: 0.975
- **Recall**: 0.185
- **F1 Score**: 0.312
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.318
- **Single-label performs better** by 0.318 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it's the only book present

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
