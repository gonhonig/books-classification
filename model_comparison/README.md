# Model Comparison Analysis

This directory contains comprehensive comparison plots and analysis for the four different book classification models:

## Models Compared

1. **Per-Book Model** (`model_per_book`) - Individual models trained for each book
2. **Multi-Label Model** (`multi_label_model`) - Single model handling all books with multi-label classification
3. **Naive Per-Book Model** (`naive_model_per_book`) - Simplified per-book models
4. **Naive Multi-Label Model** (`naive_multi_label_model`) - Simplified multi-label model

## Generated Plots

### 1. `model_accuracy_comparison.png`
Main comparison bar plot showing accuracy differences between all models with splits for:
- **Overall Accuracy**: Average performance across all books
- **Single-Label Accuracy**: Performance on single-label samples
- **Multi-Label Accuracy**: Performance on multi-label samples

### 2. `detailed_model_comparison.png`
Three subplots showing detailed comparison for each metric type:
- Overall accuracy comparison
- Single-label accuracy comparison  
- Multi-label accuracy comparison

### 3. `per_book_model_comparison.png`
Four subplots (one for each book) showing how each model performs on individual books:
- Anna Karenina
- Frankenstein
- The Adventures of Alice in Wonderland
- Wuthering Heights

Each subplot shows the three accuracy metrics (overall, single-label, multi-label) for all models.

### 4. `model_performance_heatmap.png`
Heatmap visualization showing all accuracy metrics for all models in a color-coded matrix format.

### 5. `model_performance_summary_table.png`
Formatted table showing the numerical accuracy values for all models and metric types.

## Key Insights

The plots reveal:

1. **Performance Patterns**: How different model architectures perform across different types of classification tasks
2. **Book-Specific Performance**: Which models work better for specific books
3. **Single vs Multi-Label Performance**: How models handle different types of classification challenges
4. **Overall Model Rankings**: Which approaches are most effective for the book classification task

## Usage

To regenerate the comparison plots:

```bash
cd model_comparison
python compare_all_models.py
```

This will create all the visualization plots based on the latest metrics from each model's detailed metrics JSON files.

## Data Sources

The comparison uses metrics from:
- `model_per_book/per_book_detailed_metrics.json`
- `multi_label_model/multi_label_detailed_metrics.json`
- `naive_model_per_book/naive_per_book_detailed_metrics.json`
- `naive_multi_label_model/naive_multi_label_detailed_metrics.json` 