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
| **Anna Karenina** | 0.880 | 0.879 | 0.443 | **0.920** | Multi-label excels |
| **Wuthering Heights** | 0.851 | 0.848 | 0.592 | **0.896** | Multi-label excels |
| **Frankenstein** | 0.862 | 0.847 | **0.925** | 0.709 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.895 | 0.801 | **0.873** | 0.753 | Single-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Individual Model Performance**: Each binary classifier achieves strong performance (85-89% accuracy) for its specific book identification task.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.880
- **Precision**: 0.895
- **Recall**: 0.862
- **F1 Score**: 0.879

#### Single-Label Performance
- **Accuracy**: 0.902
- **Precision**: 0.326
- **Recall**: 0.690
- **F1 Score**: 0.443
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.859
- **Precision**: 0.974
- **Recall**: 0.872
- **F1 Score**: 0.920
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.478
- **Multi-label performs better** by 0.478 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.851
- **Precision**: 0.852
- **Recall**: 0.845
- **F1 Score**: 0.848

#### Single-Label Performance
- **Accuracy**: 0.873
- **Precision**: 0.539
- **Recall**: 0.657
- **F1 Score**: 0.592
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.830
- **Precision**: 0.917
- **Recall**: 0.875
- **F1 Score**: 0.896
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.303
- **Multi-label performs better** by 0.303 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it appears alongside other books
- **Interpretation**: Wuthering Heights's distinctive gothic style and emotional intensity is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.862
- **Precision**: 0.845
- **Recall**: 0.849
- **F1 Score**: 0.847

#### Single-Label Performance
- **Accuracy**: 0.913
- **Precision**: 0.916
- **Recall**: 0.935
- **F1 Score**: 0.925
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.813
- **Precision**: 0.717
- **Recall**: 0.700
- **F1 Score**: 0.709
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.217
- **Single-label performs better** by 0.217 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.895
- **Precision**: 0.821
- **Recall**: 0.783
- **F1 Score**: 0.801

#### Single-Label Performance
- **Accuracy**: 0.945
- **Precision**: 0.916
- **Recall**: 0.834
- **F1 Score**: 0.873
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.847
- **Precision**: 0.759
- **Recall**: 0.747
- **F1 Score**: 0.753
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.120
- **Single-label performs better** by 0.120 F1 points
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
