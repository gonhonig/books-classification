# Multi-Label vs Per-Book Approach Comparison Report

## Executive Summary

This report compares two different approaches for book classification:
1. **Per-Book Approach**: Individual binary classifiers for each book
2. **Multi-Label Approach**: Single unified model for all books

## Overall Performance Comparison

| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |
|------|-------------|----------------|-----------|--------|
| Anna Karenina | 0.879 | 0.880 | +0.001 | Multi-Label |
| Wuthering Heights | 0.848 | 0.852 | +0.004 | Multi-Label |
| Frankenstein | 0.847 | 0.844 | -0.003 | Per-Book |
| The Adventures of Alice in Wonderland | 0.801 | 0.797 | -0.005 | Per-Book |

## Single-Label Performance Comparison

| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |
|------|-------------|----------------|-----------|--------|
| Anna Karenina | 0.443 | 0.469 | +0.026 | Multi-Label |
| Wuthering Heights | 0.592 | 0.608 | +0.016 | Multi-Label |
| Frankenstein | 0.925 | 0.912 | -0.013 | Per-Book |
| The Adventures of Alice in Wonderland | 0.873 | 0.866 | -0.007 | Per-Book |

## Multi-Label Performance Comparison

| Book | Per-Book F1 | Multi-Label F1 | Difference | Winner |
|------|-------------|----------------|-----------|--------|
| Anna Karenina | 0.920 | 0.918 | -0.002 | Per-Book |
| Wuthering Heights | 0.896 | 0.900 | +0.004 | Multi-Label |
| Frankenstein | 0.709 | 0.720 | +0.011 | Multi-Label |
| The Adventures of Alice in Wonderland | 0.753 | 0.747 | -0.006 | Per-Book |

## Detailed Analysis

### Average Performance

- **Overall Performance**:
  - Per-Book Approach: 0.844
  - Multi-Label Approach: 0.843
  - Difference: -0.001

- **Single-Label Performance**:
  - Per-Book Approach: 0.708
  - Multi-Label Approach: 0.714
  - Difference: +0.005

- **Multi-Label Performance**:
  - Per-Book Approach: 0.819
  - Multi-Label Approach: 0.821
  - Difference: +0.002

### Performance Patterns

- **Overall Winners**: Multi-Label wins in 2/4 books
- **Single-Label Winners**: Multi-Label wins in 2/4 books
- **Multi-Label Winners**: Multi-Label wins in 2/4 books

### Book-Specific Analysis

#### Anna Karenina

- **Overall**: Multi-Label wins by 0.001 F1 points
- **Single-Label**: Multi-Label wins by 0.026 F1 points
- **Multi-Label**: Multi-Label loses by 0.002 F1 points

**Interpretation**: The unified multi-label model performs better for Anna Karenina, suggesting that shared representations help identify this book's distinctive features.

#### Wuthering Heights

- **Overall**: Multi-Label wins by 0.004 F1 points
- **Single-Label**: Multi-Label wins by 0.016 F1 points
- **Multi-Label**: Multi-Label wins by 0.004 F1 points

**Interpretation**: The unified multi-label model performs better for Wuthering Heights, suggesting that shared representations help identify this book's distinctive features.

#### Frankenstein

- **Overall**: Multi-Label loses by 0.003 F1 points
- **Single-Label**: Multi-Label loses by 0.013 F1 points
- **Multi-Label**: Multi-Label wins by 0.011 F1 points

**Interpretation**: The specialized per-book model performs better for Frankenstein, suggesting that focused training on this specific book's characteristics is more effective.

#### The Adventures of Alice in Wonderland

- **Overall**: Multi-Label loses by 0.005 F1 points
- **Single-Label**: Multi-Label loses by 0.007 F1 points
- **Multi-Label**: Multi-Label loses by 0.006 F1 points

**Interpretation**: The specialized per-book model performs better for The Adventures of Alice in Wonderland, suggesting that focused training on this specific book's characteristics is more effective.

## Recommendations

### Primary Recommendation: Per-Book Approach

The per-book approach shows better overall performance and should be preferred for:
- **Specialized Performance**: Each model optimized for specific book
- **Better Single-Label Performance**: Excels at identifying individual books
- **Modular Design**: Independent models can be updated separately
- **Interpretability**: Clear which model is responsible for each book

### Hybrid Approach Consideration

Consider a hybrid approach based on use case:
- **Multi-Label Model**: For scenarios with mixed book content
- **Per-Book Models**: For scenarios with single book identification
- **Ensemble**: Combine both approaches for maximum accuracy

## Technical Considerations

### Multi-Label Approach
- **Pros**: Unified model, shared representations, easier deployment
- **Cons**: Complex training, potential for interference between books

### Per-Book Approach
- **Pros**: Specialized models, independent optimization, clear interpretability
- **Cons**: Multiple models to maintain, no shared learning

