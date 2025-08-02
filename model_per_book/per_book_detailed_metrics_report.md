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
| **Anna Karenina** | 0.872 | 0.872 | 0.392 | **0.922** | Multi-label excels |
| **Wuthering Heights** | 0.851 | 0.853 | 0.567 | **0.909** | Multi-label excels |
| **Frankenstein** | 0.853 | 0.831 | **0.899** | 0.709 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.889 | 0.797 | 0.848 | 0.763 | Balanced performance |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Individual Model Performance**: Each binary classifier achieves strong performance (85-89% accuracy) for its specific book identification task.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.872
- **Precision**: 0.876
- **Recall**: 0.868
- **F1 Score**: 0.872

#### Single-Label Performance
- **Accuracy**: 0.884
- **Precision**: 0.277
- **Recall**: 0.667
- **F1 Score**: 0.392
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.860
- **Precision**: 0.968
- **Recall**: 0.879
- **F1 Score**: 0.922
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.530
- **Multi-label performs better** by 0.530 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.851
- **Precision**: 0.827
- **Recall**: 0.881
- **F1 Score**: 0.853

#### Single-Label Performance
- **Accuracy**: 0.853
- **Precision**: 0.483
- **Recall**: 0.686
- **F1 Score**: 0.567
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.848
- **Precision**: 0.906
- **Recall**: 0.913
- **F1 Score**: 0.909
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.342
- **Multi-label performs better** by 0.342 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it appears alongside other books
- **Interpretation**: Wuthering Heights's distinctive gothic style and emotional intensity is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.853
- **Precision**: 0.856
- **Recall**: 0.806
- **F1 Score**: 0.831

#### Single-Label Performance
- **Accuracy**: 0.886
- **Precision**: 0.918
- **Recall**: 0.882
- **F1 Score**: 0.899
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.820
- **Precision**: 0.744
- **Recall**: 0.676
- **F1 Score**: 0.709
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.191
- **Single-label performs better** by 0.191 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.889
- **Precision**: 0.785
- **Recall**: 0.810
- **F1 Score**: 0.797

#### Single-Label Performance
- **Accuracy**: 0.933
- **Precision**: 0.870
- **Recall**: 0.828
- **F1 Score**: 0.848
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.846
- **Precision**: 0.733
- **Recall**: 0.797
- **F1 Score**: 0.763
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.085
- **Single-label performs better** by 0.085 F1 points
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
3. **Overall**: All individual models achieve strong performance (85-89% accuracy) for individual book identification

## Methodology Notes

- **Model Architecture**: Individual binary classifiers for each book
- **Training Data**: Pre-existing dataset splits with aligned embeddings
- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis
- **Threshold**: 0.5 probability threshold for binary classification
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation
- **Approach**: Separate specialized model for each book, optimized for individual book identification
