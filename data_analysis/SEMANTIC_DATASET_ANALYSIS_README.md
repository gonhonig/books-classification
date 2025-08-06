# Semantic Augmented Dataset Analysis

This directory contains scripts for analyzing the semantic augmented dataset and visualizing the embeddings space.

## Scripts

### `semantic_dataset_analysis.py`
Comprehensive analysis script that:
- Loads the semantic augmented dataset (excluding `naive_` columns)
- Loads cached embeddings
- Performs detailed statistical analysis of the dataset
- Analyzes embedding quality with separation ratios
- Creates advanced visualizations including PCA and t-SNE
- Focuses analysis on `original_book` (since it's equivalent to `original_label`)
- Saves all results to JSON file
- Generates comprehensive plots

## Dataset Structure

The semantic augmented dataset contains:
- **Total samples**: 10,120
- **Columns**: 7 (excluding `naive_` columns)
  - `sentence`: The text content
  - `original_label`: Original classification label (0-3) - equivalent to book mapping
  - `original_book`: Source book name (primary analysis focus)
  - `book_Anna_Karenina`: Multi-label classification
  - `book_Frankenstein`: Multi-label classification
  - `book_The_Adventures_of_Alice_in_Wonderland`: Multi-label classification
  - `book_Wuthering_Heights`: Multi-label classification

## Embeddings

- **Dimensions**: 384
- **Source**: Cached embeddings from sentence transformers
- **Files**: 
  - `../data/embeddings_cache_aligned_f24a423ed8f9dd531230fe64f71f668d.npz` (primary)
  - `../data/embeddings_cache_4bdc0800b2ede390f133eed833a83211.npz` (fallback)

## Key Findings

### Dataset Statistics
- **Book distribution**: Frankenstein (32.3%) and Anna Karenina (30.6%) are most represented
- **Average sentence length**: 83.6 characters
- **Multi-label patterns**: Most samples have 1-2 labels, with some having 3-4 labels

### Embedding Quality
- **Average embedding norm**: 4.6110
- **Separation ratios**: All close to 1.0, indicating similar within-book and between-book distances
- **PCA explained variance**: ~12.2% for 2D, ~16.4% for 3D

### Visualization Results
- **PCA plots**: Show clustering by books and multi-label counts
- **t-SNE plots**: Better separation, especially for book-based clustering
- **3D visualizations**: Provide additional dimensionality for analysis

## Generated Files

### Visualizations:
- `semantic_dataset_comprehensive_analysis.png`: 2x2 grid with PCA and t-SNE plots
  - PCA by Original Book
  - PCA by Multi-label Count
  - t-SNE by Original Book (Subset)
  - t-SNE by Multi-label Count (Subset)
- `semantic_dataset_3d_pca_detailed.png`: Detailed 3D PCA plot by book
- `semantic_dataset_analysis_results.json`: Complete analysis results in JSON format

## Usage

```bash
# Navigate to data_analysis directory
cd data_analysis

# Run the analysis
python semantic_dataset_analysis.py
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- warnings (built-in)

## Analysis Focus

The analysis focuses on `original_book` rather than `original_label` since they represent the same information:
- Label 0 → The Adventures of Alice in Wonderland
- Label 1 → Anna Karenina  
- Label 2 → Wuthering Heights
- Label 3 → Frankenstein

This simplification provides clearer insights into the semantic structure of the dataset.

## Notes

- The script automatically excludes columns starting with "naive_" as requested
- Embeddings are loaded from cached files for efficiency
- t-SNE is computed on a subset (2000 samples) for performance
- All visualizations are saved as high-resolution PNG files
- Analysis results are saved in structured JSON format for further processing
- The script uses relative paths to access data from the parent directory 