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
| **Anna Karenina** | 0.881 | 0.879 | 0.393 | **0.922** | Multi-label excels |
| **Frankenstein** | 0.848 | 0.829 | **0.911** | 0.678 | Single-label excels |
| **The Adventures of Alice in Wonderland** | 0.877 | 0.748 | **0.864** | 0.662 | Single-label excels |
| **Wuthering Heights** | 0.845 | 0.845 | 0.597 | **0.894** | Multi-label excels |
## Per-Book Performance

### Anna Karenina

#### Overall Performance
- **Accuracy**: 0.881
- **Precision**: 0.899
- **Recall**: 0.860
- **F1 Score**: 0.879

#### Single-Label Performance
- **Accuracy**: 0.901
- **Precision**: 0.300
- **Recall**: 0.571
- **F1 Score**: 0.393
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.861
- **Precision**: 0.972
- **Recall**: 0.877
- **F1 Score**: 0.922
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.529
---

### Frankenstein

#### Overall Performance
- **Accuracy**: 0.848
- **Precision**: 0.837
- **Recall**: 0.821
- **F1 Score**: 0.829

#### Single-Label Performance
- **Accuracy**: 0.896
- **Precision**: 0.898
- **Recall**: 0.923
- **F1 Score**: 0.911
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.802
- **Precision**: 0.716
- **Recall**: 0.644
- **F1 Score**: 0.678
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.233
---

### The Adventures of Alice in Wonderland

#### Overall Performance
- **Accuracy**: 0.877
- **Precision**: 0.835
- **Recall**: 0.678
- **F1 Score**: 0.748

#### Single-Label Performance
- **Accuracy**: 0.942
- **Precision**: 0.926
- **Recall**: 0.811
- **F1 Score**: 0.864
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.813
- **Precision**: 0.762
- **Recall**: 0.585
- **F1 Score**: 0.662
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: -0.202
---

### Wuthering Heights

#### Overall Performance
- **Accuracy**: 0.845
- **Precision**: 0.832
- **Recall**: 0.858
- **F1 Score**: 0.845

#### Single-Label Performance
- **Accuracy**: 0.866
- **Precision**: 0.517
- **Recall**: 0.705
- **F1 Score**: 0.597
- **Sample Count**: 747

#### Multi-Label Performance
- **Accuracy**: 0.825
- **Precision**: 0.904
- **Recall**: 0.883
- **F1 Score**: 0.894
- **Sample Count**: 772

#### Performance Analysis
- **F1 Difference (Multi - Single)**: +0.297
---

