# Books Classification Dataset

This directory contains the processed datasets for the English book classification project, which classifies sentences from four classic novels into their respective books.

## 📚 Books in the Dataset

The dataset includes sentences from four classic English novels:

1. **Anna Karenina** by Leo Tolstoy
2. **The Adventures of Alice in Wonderland** by Lewis Carroll
3. **Frankenstein** by Mary Shelley
4. **Wuthering Heights** by Emily Brontë

## 📁 Dataset Files

### Core Corpus Files

- **`corpus.json`**: Original corpus with all sentences from the books
  - **Total sentences**: 32,757
  - **Format**: JSON with book titles as keys
  - **Structure**: Each book contains `title`, `author`, `genre`, and `sentences` array

- **`corpus_deduplicated.json`**: Deduplicated corpus with duplicate sentences removed
  - **Total sentences**: 31,760 (997 duplicates removed, 3.0% reduction)
  - **Format**: JSON with same structure as original corpus
  - **Purpose**: Clean dataset for training

### Processed Datasets

- **`full_dataset.csv`**: Complete dataset with all sentences and labels
  - **Total sentences**: 31,223 (after preprocessing)
  - **Format**: CSV with columns: `sentence`, `label`, `book_id`
  - **Purpose**: Base dataset without balancing or splitting

- **`semantic_augmented/semantic_augmented_dataset.csv`**: Augmented dataset with multi-label annotations
  - **Total sentences**: 31,760
  - **Format**: CSV with book columns (0/1 labels)
  - **Purpose**: Dataset with semantic similarity augmentations

- **`semantic_augmented/semantic_augmented_balanced_dataset.csv`**: Balanced dataset for training
  - **Total sentences**: 10,120
  - **Format**: CSV with balanced multi-label annotations
  - **Purpose**: Training dataset with balanced class distribution

### Dataset Splits

- **`dataset/`**: HuggingFace DatasetDict with train/validation/test splits
  - **Train**: 7,083 samples
  - **Validation**: 1,518 samples
  - **Test**: 1,519 samples
  - **Format**: Arrow files with multi-label format

### Metadata Files

- **`metadata.json`**: Basic dataset metadata
- **`semantic_augmented/dataset_metadata.json`**: Comprehensive statistics
- **`semantic_augmented/dataset_metadata_summary.md`**: Human-readable summary

## 📊 Dataset Statistics

### Original Corpus Distribution

| Book | Original Sentences | Deduplicated Sentences | Removed | % Reduction |
|------|-------------------|----------------------|---------|-------------|
| Anna Karenina | 21,050 | 20,344 | 706 | 3.4% |
| Wuthering Heights | 6,766 | 6,583 | 183 | 2.7% |
| Frankenstein | 3,316 | 3,269 | 47 | 1.4% |
| Alice in Wonderland | 1,625 | 1,564 | 61 | 3.8% |
| **Total** | **32,757** | **31,760** | **997** | **3.0%** |

### Balanced Dataset Statistics

- **Total samples**: 10,120
- **Multi-label samples**: 5,154 (51%)
- **Single-label samples**: 4,966 (49%)

#### Label Distribution
- **1-label**: 4,966 samples (49%)
- **2-labels**: 3,479 samples (34%)
- **3-labels**: 1,264 samples (12%)
- **4-labels**: 411 samples (4%)

#### Book Distribution in Balanced Dataset
- **Anna Karenina**: 5,000 positive, 5,120 negative
- **Wuthering Heights**: 5,000 positive, 5,120 negative
- **Frankenstein**: 4,443 positive, 5,677 negative
- **Alice in Wonderland**: 2,917 positive, 7,203 negative

### Split Distribution

| Split | Samples | % of Total |
|-------|---------|------------|
| Train | 7,083 | 70% |
| Validation | 1,518 | 15% |
| Test | 1,519 | 15% |

## 🔧 Dataset Creation Process

### 1. Data Collection
- Downloaded from HuggingFace dataset: `IsmaelMousa/books`
- Filtered for the four selected books
- Extracted English text and split into sentences

### 2. Preprocessing
- **Text normalization**: Removed extra whitespace, normalized line breaks
- **Length filtering**: Kept sentences between 10-512 characters
- **Deduplication**: Removed duplicate sentences within each book
- **Special character handling**: Optional removal based on config

### 3. Semantic Augmentation
- **Similarity detection**: Used semantic embeddings to find similar sentences
- **Multi-label creation**: Sentences similar across books get multiple labels
- **Threshold**: 0.7 similarity threshold for augmentation

### 4. Balancing Strategy
- **Multi-label priority**: All multi-label sentences included
- **Single-label balancing**: Added single-label samples to reach target (5,000 per class)
- **Negative balancing**: Added negative samples to balance positive/negative ratios
- **Final dataset**: 10,120 samples with balanced distribution

## 📈 Usage Examples

### Loading the Full Dataset
```python
import pandas as pd

# Load the full dataset
df = pd.read_csv('data/full_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Books: {df['book_id'].unique()}")
```

### Loading the Balanced Dataset
```python
# Load the balanced dataset
balanced_df = pd.read_csv('data/semantic_augmented/semantic_augmented_balanced_dataset.csv')
print(f"Balanced dataset shape: {balanced_df.shape}")

# Check multi-label distribution
book_columns = [col for col in balanced_df.columns if col.startswith('book_')]
multi_label_count = (balanced_df[book_columns].sum(axis=1) > 1).sum()
print(f"Multi-label samples: {multi_label_count}")
```

### Loading Dataset Splits
```python
from datasets import load_from_disk

# Load the dataset splits
dataset = load_from_disk('data/dataset')
print(f"Train: {len(dataset['train'])}")
print(f"Validation: {len(dataset['validation'])}")
print(f"Test: {len(dataset['test'])}")
```

### Loading Metadata
```python
import json

# Load comprehensive metadata
with open('data/semantic_augmented/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Total samples: {metadata['total_size']}")
print(f"Label distribution: {metadata['label_distribution']}")
```

## 🎯 Dataset Characteristics

### Multi-label Classification
- **Task**: Classify sentences into one or more books
- **Classes**: 4 books (Anna Karenina, Alice in Wonderland, Frankenstein, Wuthering Heights)
- **Labels**: Binary (0/1) for each book
- **Multi-label**: Sentences can belong to multiple books

### Semantic Similarity
- **Augmentation**: Uses semantic embeddings to find similar sentences across books
- **Threshold**: 0.7 similarity for multi-label creation
- **Benefit**: Captures thematic similarities and shared literary elements

### Balanced Training
- **Strategy**: Prioritizes multi-label sentences while maintaining class balance
- **Target**: 5,000 samples per class where possible
- **Distribution**: Balanced positive/negative samples for each book

## 📋 File Descriptions

| File | Description | Size | Format |
|------|-------------|------|--------|
| `corpus.json` | Original corpus with all sentences | 32,757 sentences | JSON |
| `corpus_deduplicated.json` | Cleaned corpus without duplicates | 31,760 sentences | JSON |
| `full_dataset.csv` | Complete dataset with labels | 31,223 sentences | CSV |
| `semantic_augmented_dataset.csv` | Augmented with multi-labels | 31,760 sentences | CSV |
| `semantic_augmented_balanced_dataset.csv` | Balanced for training | 10,120 sentences | CSV |
| `dataset/` | Train/val/test splits | 10,120 sentences | Arrow |
| `metadata.json` | Basic dataset info | - | JSON |
| `dataset_metadata.json` | Comprehensive statistics | - | JSON |

## 🔄 Data Pipeline

```
Raw Books → Corpus Creation → Deduplication → Semantic Augmentation → Balancing → Splitting
    ↓              ↓              ↓              ↓              ↓           ↓
corpus.json → corpus_deduplicated.json → semantic_augmented_dataset.csv → balanced_dataset.csv → dataset/
```

## 📝 Notes

- **Reproducibility**: All random operations use seed 42
- **Memory efficient**: Uses HuggingFace datasets for large files
- **Flexible**: Multiple dataset formats for different use cases
- **Comprehensive**: Detailed metadata and statistics for analysis

## 🤝 Contributing

When modifying the dataset:
1. Update this README with new statistics
2. Regenerate metadata files
3. Document any changes to the balancing strategy
4. Update the data pipeline description

---

*Last updated: August 2024*
*Dataset version: 1.0*
*Total processing time: ~5 minutes* 