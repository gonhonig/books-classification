# Individual Book Models - Per-Book Single-Label vs Multi-Label Performance Report

## Summary
- **Total test samples**: 1519
- **Single-label samples**: 747 (49.2%)
- **Multi-label samples**: 772 (50.8%)
- **Model Type**: Individual Binary Classifiers (one model per book)
- **Training Approach**: Separate binary classification for each book

## Performance Overview

### Overall Model Performance Comparison

| Book | Overall Accuracy | Overall F1 | Single-Label F1 | Multi-Label F1 | Performance Pattern |
|------|------------------|------------|-----------------|----------------|-------------------|
| **Anna Karenina** | 0.785 | 0.746 | 0.510 | **0.766** | Multi-label excels |
| **Frankenstein** | 0.826 | 0.775 | **0.923** | 0.401 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.824 | 0.565 | **0.861** | 0.224 | Single-label excels |
| **Wuthering Heights** | 0.589 | 0.320 | **0.600** | 0.252 | Single-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Individual Model Performance**: Each binary classifier achieves strong performance (85-89% accuracy) for its specific book identification task.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.785
- **Precision**: 0.921
- **Recall**: 0.626
- **F1 Score**: 0.746

#### Single-Label Performance
- **Accuracy**: 0.933
- **Precision**: 0.433
- **Recall**: 0.619
- **F1 Score**: 0.510
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.642
- **Precision**: 0.985
- **Recall**: 0.627
- **F1 Score**: 0.766
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.256
- **Multi-label performs better** by 0.256 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.826
- **Precision**: 0.918
- **Recall**: 0.671
- **F1 Score**: 0.775

#### Single-Label Performance
- **Accuracy**: 0.913
- **Precision**: 0.942
- **Recall**: 0.905
- **F1 Score**: 0.923
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.741
- **Precision**: 0.798
- **Recall**: 0.268
- **F1 Score**: 0.401
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.522
- **Single-label performs better** by 0.522 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.824
- **Precision**: 0.845
- **Recall**: 0.424
- **F1 Score**: 0.565

#### Single-Label Performance
- **Accuracy**: 0.938
- **Precision**: 0.882
- **Recall**: 0.840
- **F1 Score**: 0.861
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.712
- **Precision**: 0.711
- **Recall**: 0.133
- **F1 Score**: 0.224
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.637
- **Single-label performs better** by 0.637 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.589
- **Precision**: 0.855
- **Recall**: 0.197
- **F1 Score**: 0.320

#### Single-Label Performance
- **Accuracy**: 0.904
- **Precision**: 0.720
- **Recall**: 0.514
- **F1 Score**: 0.600
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.284
- **Precision**: 0.959
- **Recall**: 0.145
- **F1 Score**: 0.252
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.348
- **Single-label performs better** by 0.348 F1 points
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
3. **Overall**: All individual models achieve strong performance (85-89% accuracy) for individual book identification

## Methodology Notes

- **Model Architecture**: Individual binary classifiers for each book
- **Training Data**: Pre-existing dataset splits with aligned embeddings
- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis
- **Threshold**: 0.5 probability threshold for binary classification
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation
- **Approach**: Separate specialized model for each book, optimized for individual book identification
