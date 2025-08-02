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
| **Anna Karenina** | 0.882 | 0.882 | 0.433 | **0.926** | Multi-label excels |
| **Wuthering Heights** | 0.860 | 0.863 | 0.581 | **0.921** | Multi-label excels |
| **Frankenstein** | 0.849 | 0.827 | **0.923** | 0.648 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.883 | 0.781 | **0.862** | 0.727 | Single-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.882
- **Precision**: 0.885
- **Recall**: 0.879
- **F1 Score**: 0.882

#### Single-Label Performance
- **Accuracy**: 0.898
- **Precision**: 0.315
- **Recall**: 0.690
- **F1 Score**: 0.433
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.867
- **Precision**: 0.964
- **Recall**: 0.890
- **F1 Score**: 0.926
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.493
- **Multi-label performs better** by 0.493 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.860
- **Precision**: 0.830
- **Recall**: 0.898
- **F1 Score**: 0.863

#### Single-Label Performance
- **Accuracy**: 0.851
- **Precision**: 0.481
- **Recall**: 0.733
- **F1 Score**: 0.581
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.868
- **Precision**: 0.917
- **Recall**: 0.925
- **F1 Score**: 0.921
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.340
- **Multi-label performs better** by 0.340 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it appears alongside other books
- **Interpretation**: Wuthering Heights's distinctive gothic style and emotional intensity is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.849
- **Precision**: 0.851
- **Recall**: 0.805
- **F1 Score**: 0.827

#### Single-Label Performance
- **Accuracy**: 0.910
- **Precision**: 0.919
- **Recall**: 0.926
- **F1 Score**: 0.923
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.790
- **Precision**: 0.710
- **Recall**: 0.596
- **F1 Score**: 0.648
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.275
- **Single-label performs better** by 0.275 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.883
- **Precision**: 0.789
- **Recall**: 0.773
- **F1 Score**: 0.781

#### Single-Label Performance
- **Accuracy**: 0.940
- **Precision**: 0.897
- **Recall**: 0.828
- **F1 Score**: 0.862
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.828
- **Precision**: 0.720
- **Recall**: 0.734
- **F1 Score**: 0.727
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.135
- **Single-label performs better** by 0.135 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

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
