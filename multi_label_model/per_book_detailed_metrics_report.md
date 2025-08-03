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
| **Anna Karenina** | 0.882 | 0.880 | 0.469 | **0.918** | Multi-label excels |
| **Wuthering Heights** | 0.853 | 0.852 | 0.608 | **0.900** | Multi-label excels |
| **Frankenstein** | 0.862 | 0.844 | **0.912** | 0.720 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.891 | 0.797 | **0.866** | 0.747 | Single-label excels |

### Key Insights

1. **Multi-Label Specialists**: Books that excel at multi-label classification have distinctive writing styles that become more apparent when contrasted with other books.

2. **Single-Label Specialists**: Books that excel at single-label classification have very distinctive styles that are easily recognizable in isolation.

3. **Unified Model Performance**: The multi-label neural network achieves strong per-book performance (84-88% accuracy) while handling all books simultaneously.

## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.882
- **Precision**: 0.898
- **Recall**: 0.862
- **F1 Score**: 0.880

#### Single-Label Performance
- **Accuracy**: 0.909
- **Precision**: 0.349
- **Recall**: 0.714
- **F1 Score**: 0.469
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.855
- **Precision**: 0.971
- **Recall**: 0.871
- **F1 Score**: 0.918
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.449
- **Multi-label performs better** by 0.449 F1 points
- **Pattern**: This model excels at identifying Anna Karenina when it appears alongside other books
- **Interpretation**: Anna Karenina's distinctive writing style (Tolstoy's detailed character development and social commentary) is more recognizable in multi-label contexts

---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.853
- **Precision**: 0.843
- **Recall**: 0.861
- **F1 Score**: 0.852

#### Single-Label Performance
- **Accuracy**: 0.869
- **Precision**: 0.524
- **Recall**: 0.724
- **F1 Score**: 0.608
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.837
- **Precision**: 0.917
- **Recall**: 0.883
- **F1 Score**: 0.900
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.292
- **Multi-label performs better** by 0.292 F1 points
- **Pattern**: This model excels at identifying Wuthering Heights when it appears alongside other books
- **Interpretation**: Wuthering Heights's distinctive gothic style and emotional intensity is more recognizable in multi-label contexts

---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.862
- **Precision**: 0.855
- **Recall**: 0.833
- **F1 Score**: 0.844

#### Single-Label Performance
- **Accuracy**: 0.898
- **Precision**: 0.908
- **Recall**: 0.916
- **F1 Score**: 0.912
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.826
- **Precision**: 0.754
- **Recall**: 0.688
- **F1 Score**: 0.720
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.193
- **Single-label performs better** by 0.193 F1 points
- **Pattern**: This model excels at identifying Frankenstein when it's the only book present
- **Interpretation**: Frankenstein's distinctive gothic horror and scientific themes are very recognizable in isolation

---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.891
- **Precision**: 0.805
- **Recall**: 0.788
- **F1 Score**: 0.797

#### Single-Label Performance
- **Accuracy**: 0.940
- **Precision**: 0.869
- **Recall**: 0.864
- **F1 Score**: 0.866
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.845
- **Precision**: 0.760
- **Recall**: 0.734
- **F1 Score**: 0.747
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
3. **Overall**: The unified multi-label model achieves strong performance (84-88% accuracy) for individual book identification

## Methodology Notes

- **Model Architecture**: Unified multi-label neural network with sigmoid outputs
- **Training Data**: Pre-existing dataset splits with aligned embeddings
- **Evaluation**: Per-book metrics calculated on test set with single-label vs multi-label analysis
- **Threshold**: 0.5 probability threshold for binary classification
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for comprehensive evaluation
- **Approach**: Single model handles all books simultaneously, leveraging shared representations
