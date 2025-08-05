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
| **Anna Karenina** | 0.884 | 0.885 | 0.403 | **0.933** | Multi-label excels |
| **Frankenstein** | 0.857 | 0.840 | **0.910** | 0.719 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.895 | 0.802 | 0.853 | 0.770 | Balanced performance |
| **Wuthering Heights** | 0.849 | 0.851 | 0.586 | **0.906** | Multi-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Individual Model Performance**: Each binary classifier achieves strong performance (85-89% accuracy) for its specific book identification task.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.884
- **Precision**: 0.883
- **Recall**: 0.887
- **F1 Score**: 0.885

#### Single-Label Performance
- **Accuracy**: 0.889
- **Precision**: 0.289
- **Recall**: 0.667
- **F1 Score**: 0.403
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.880
- **Precision**: 0.969
- **Recall**: 0.900
- **F1 Score**: 0.933
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.530
- **Multi-label performs better** by 0.530 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.857
- **Precision**: 0.843
- **Recall**: 0.837
- **F1 Score**: 0.840

#### Single-Label Performance
- **Accuracy**: 0.897
- **Precision**: 0.914
- **Recall**: 0.907
- **F1 Score**: 0.910
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.819
- **Precision**: 0.722
- **Recall**: 0.716
- **F1 Score**: 0.719
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.191
- **Single-label performs better** by 0.191 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.895
- **Precision**: 0.812
- **Recall**: 0.793
- **F1 Score**: 0.802

#### Single-Label Performance
- **Accuracy**: 0.937
- **Precision**: 0.907
- **Recall**: 0.805
- **F1 Score**: 0.853
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.854
- **Precision**: 0.756
- **Recall**: 0.784
- **F1 Score**: 0.770
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.083
- **Single-label performs better** by 0.083 F1 points
- **Pattern**: This model excels at identifying The Adventures of Alice in Wonderland when it's the only book present
- **Interpretation**: The Adventures of Alice in Wonderland's distinctive whimsical and fantastical style is very recognizable in isolation

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.849
- **Precision**: 0.825
- **Recall**: 0.878
- **F1 Score**: 0.851

#### Single-Label Performance
- **Accuracy**: 0.853
- **Precision**: 0.484
- **Recall**: 0.743
- **F1 Score**: 0.586
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.845
- **Precision**: 0.912
- **Recall**: 0.900
- **F1 Score**: 0.906
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.319
- **Multi-label performs better** by 0.319 F1 points
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
