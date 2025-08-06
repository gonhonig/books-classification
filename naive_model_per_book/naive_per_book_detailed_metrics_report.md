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
---

