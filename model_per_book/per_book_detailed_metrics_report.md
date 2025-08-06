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
---

