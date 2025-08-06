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
| **Anna Karenina** | 0.793 | 0.760 | 0.518 | **0.783** | Multi-label excels |
| **Frankenstein** | 0.805 | 0.737 | **0.894** | 0.327 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.826 | 0.545 | **0.856** | 0.185 | Single-label excels |
| **Wuthering Heights** | 0.589 | 0.343 | **0.635** | 0.261 | Single-label excels |
## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.793
- **Precision**: 0.912
- **Recall**: 0.651
- **F1 Score**: 0.760

#### Single-Label Performance
- **Accuracy**: 0.928
- **Precision**: 0.414
- **Recall**: 0.690
- **F1 Score**: 0.518
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.663
- **Precision**: 0.985
- **Recall**: 0.649
- **F1 Score**: 0.783
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.265
---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.805
- **Precision**: 0.935
- **Recall**: 0.608
- **F1 Score**: 0.737

#### Single-Label Performance
- **Accuracy**: 0.885
- **Precision**: 0.953
- **Recall**: 0.842
- **F1 Score**: 0.894
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.728
- **Precision**: 0.823
- **Recall**: 0.204
- **F1 Score**: 0.327
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.567
---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.826
- **Precision**: 0.919
- **Recall**: 0.388
- **F1 Score**: 0.545

#### Single-Label Performance
- **Accuracy**: 0.940
- **Precision**: 0.931
- **Recall**: 0.793
- **F1 Score**: 0.856
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.715
- **Precision**: 0.862
- **Recall**: 0.104
- **F1 Score**: 0.185
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.671
---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.589
- **Precision**: 0.803
- **Recall**: 0.218
- **F1 Score**: 0.343

#### Single-Label Performance
- **Accuracy**: 0.898
- **Precision**: 0.641
- **Recall**: 0.629
- **F1 Score**: 0.635
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.290
- **Precision**: 0.970
- **Recall**: 0.151
- **F1 Score**: 0.261
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.373
---

