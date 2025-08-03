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
| **Anna Karenina** | 0.862 | 0.854 | 0.389 | **0.892** | Multi-label excels |
| **Frankenstein** | 0.873 | 0.851 | **0.920** | 0.722 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.891 | 0.781 | **0.863** | 0.721 | Single-label excels |
| **Wuthering Heights** | 0.847 | 0.853 | 0.564 | **0.913** | Multi-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Individual Model Performance**: Each binary classifier achieves strong performance (85-89% accuracy) for its specific book identification task.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.862
- **Precision**: 0.913
- **Recall**: 0.802
- **F1 Score**: 0.854

#### Single-Label Performance
- **Accuracy**: 0.912
- **Precision**: 0.318
- **Recall**: 0.500
- **F1 Score**: 0.389
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.815
- **Precision**: 0.978
- **Recall**: 0.820
- **F1 Score**: 0.892
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.503
- **Multi-label performs better** by 0.503 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.873
- **Precision**: 0.896
- **Recall**: 0.811
- **F1 Score**: 0.851

#### Single-Label Performance
- **Accuracy**: 0.909
- **Precision**: 0.935
- **Recall**: 0.905
- **F1 Score**: 0.920
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.838
- **Precision**: 0.814
- **Recall**: 0.648
- **F1 Score**: 0.722
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.198
- **Single-label performs better** by 0.198 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.891
- **Precision**: 0.857
- **Recall**: 0.717
- **F1 Score**: 0.781

#### Single-Label Performance
- **Accuracy**: 0.942
- **Precision**: 0.932
- **Recall**: 0.805
- **F1 Score**: 0.863
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.842
- **Precision**: 0.802
- **Recall**: 0.656
- **F1 Score**: 0.721
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.142
- **Single-label performs better** by 0.142 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.847
- **Precision**: 0.811
- **Recall**: 0.900
- **F1 Score**: 0.853

#### Single-Label Performance
- **Accuracy**: 0.841
- **Precision**: 0.458
- **Recall**: 0.733
- **F1 Score**: 0.564
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.854
- **Precision**: 0.900
- **Recall**: 0.927
- **F1 Score**: 0.913
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.349
- **Multi-label performs better** by 0.349 F1 points
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
3. **Overall**: All individual models achieve strong performance (85-89% accuracy) for individual book identification

## Methodology Notes

- **Model Architecture**: Individual binary classifiers for each book
- **Training Data**: Pre-existing dataset splits with aligned embeddings
- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis
- **Threshold**: 0.5 probability threshold for binary classification
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation
- **Approach**: Separate specialized model for each book, optimized for individual book identification
