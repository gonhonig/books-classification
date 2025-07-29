# Semantic Book Classification

A project for creating semantic embedding spaces for multi-label book sentence classification, where sentences can belong to multiple books based on semantic similarity.

## Project Overview

This project implements a sophisticated approach to book sentence classification that goes beyond traditional single-label classification. Instead, it creates a semantic embedding space where:

- **Generic sentences** (like "I told him to leave") can belong to multiple books
- **Book-specific sentences** (like "The monster approached with yellow eyes") belong primarily to one book
- **Semantic similarity** determines sentence belonging rather than just source book

## Key Features

### 🎯 **Semantic Embedding Space**
- Uses pre-trained `all-MiniLM-L6-v2` model for fast, effective sentence embeddings
- Creates 384-dimensional semantic representations
- Identifies cross-book semantic similarities automatically

### 📊 **Multi-Label Classification**
- Each sentence gets a **belonging score** (0-1) for each book
- Generic sentences: high scores for multiple books
- Specific sentences: high score for one book, low for others

### 🔍 **Automated Semantic Analysis**
- **Cross-book similarity detection**: Finds semantically similar sentences across books
- **Book specificity analysis**: Identifies book-specific vs generic sentences
- **Training signal generation**: Creates multi-label targets based on semantic similarity

## Data Preparation

The data preparation pipeline:

1. **Downloads** 4 selected books from Hugging Face (`IsmaelMousa/books`)
2. **Extracts sentences** with configurable limits for testing
3. **Performs semantic analysis** using pre-trained models
4. **Generates multi-label training signals** based on cross-book similarities
5. **Creates datasets** for semantic embedding training

## GPU Training Support

The semantic embedding model supports multiple training devices:

- **CPU**: Universal compatibility, slower training
- **MPS (Apple Silicon)**: Fast training on Mac with Apple Silicon
- **CUDA**: Fastest training on NVIDIA GPUs (Linux/Windows)

Training automatically detects available devices and can be configured via command-line arguments.

### Smart Recreation System

The data preparation script includes intelligent file management:

- **Config change detection**: Automatically detects when configuration changes require file recreation
- **File existence checking**: Only recreates missing files
- **Force recreation**: Use `--force` flag to recreate all files regardless of state
- **Config hash tracking**: Stores hash of relevant config sections to detect changes

This ensures efficient development workflow - files are only recreated when necessary.

### Selected Books
- **Anna Karenina** (Romance/Drama)
- **The Adventures of Alice in Wonderland** (Fantasy/Children's)
- **Frankenstein** (Gothic/Horror)
- **The Life of Julius Caesar** (Historical/Biography)

## Configuration

The project uses a comprehensive YAML configuration (`configs/config.yaml`) that includes:

- **Data settings**: Book selection, sentence limits, splits
- **Model configuration**: Encoder type, embedding dimensions, training phases
- **Semantic analysis**: Similarity thresholds, specificity analysis
- **Training parameters**: Batch sizes, learning rates, loss weights
- **Evaluation metrics**: Multi-label metrics, visualization settings

## Current Status

✅ **Completed:**
- Data preparation pipeline with semantic analysis
- Cross-book similarity detection (29 similar pairs found)
- Multi-label training signal generation (3,679 signals created)
- Book specificity analysis
- Configuration updates for semantic approach
- **Contrastive learning model implementation**
- **Semantic embedding training (10 epochs)**
- **Model evaluation showing 5.2% improvement in similarity**
- **Embedding space visualization and analysis**

🔄 **Next Steps:**
- Create multi-label classification head
- Train joint model (embedding + classification)
- Add dimensionality reduction visualization
- Implement semantic similarity evaluation metrics

## Usage

### Data Preparation
```bash
# Basic usage - only recreates if config changed or files missing
python data/prepare_data.py

# Force recreation of all files
python data/prepare_data.py --force

# Use custom config file
python data/prepare_data.py --config path/to/config.yaml

# Show help
python data/prepare_data.py --help
```

### Semantic Analysis
```bash
python utils/semantic_analysis.py
```

### Train Semantic Embedding Model
```bash
# Train on CPU
python train_semantic_embedding.py --device cpu

# Train on GPU (MPS for Mac, CUDA for Linux/Windows)
python train_semantic_embedding.py --device mps

# Train on CUDA (if available)
python train_semantic_embedding.py --device cuda





# Comprehensive visualization (all modes)
python comprehensive_visualization.py --mode all --model improved

# Specific visualization modes
python comprehensive_visualization.py --mode basic --model improved
python comprehensive_visualization.py --mode specificity --model improved
python comprehensive_visualization.py --mode comparison --model improved

# Train improved semantic embedding model (better training data)
python train_improved_semantic_embedding.py --device mps
```

## Project Structure

```
books-classification/
├── configs/
│   └── config.yaml              # Project configuration
├── data/
│   ├── prepare_data.py          # Data preparation pipeline
│   ├── processed_dataset/       # Main dataset splits
│   ├── semantic_analysis_data.json  # Multi-label training signals
│   └── metadata.json           # Dataset metadata
├── utils/
│   └── semantic_analysis.py    # Semantic analysis utilities
└── README.md                   # This file
```

## Technical Details

### Semantic Analysis Results
- **Total sentences**: 3,679
- **Cross-book similar pairs**: 29 (similarity ≥ 0.7)
- **Embedding dimension**: 384 (all-MiniLM-L6-v2)
- **Training signals**: 3,679 multi-label examples

### Example Similar Pairs Found
- "Let me alone!" (Alice) ↔ "Ah, let me alone, let me alone!" (Anna Karenina)
- "How do you mean?" (Anna Karenina) ↔ "What do you mean?" (Julius Caesar)
- "she thought." (Alice) ↔ "she thought." (Anna Karenina) - Perfect match

This approach enables a more nuanced understanding of book content, where semantic meaning drives classification rather than simple source attribution. 