
# Individual Book Model Training Results

## Overview
- **Total Models Trained**: 4
- **Training Approach**: 4 separate binary classifiers, one for each book
- **Dataset**: Semantic augmented dataset with balanced sampling
- **Model Architecture**: Neural network with 384 input dimensions

## Performance Summary

### Average Performance Across All Models
- **Average Accuracy**: 0.8511
- **Average Precision**: 0.8553
- **Average Recall**: 0.8475
- **Average F1 Score**: 0.8510

### Best and Worst Performing Models
- **Best Model**: The Adventures of Alice in Wonderland (F1: 0.8931)
- **Worst Model**: Wuthering Heights (F1: 0.8152)

## Individual Model Results

                                 Book  Accuracy  Precision   Recall  F1 Score
                        Anna Karenina  0.862000   0.886494 0.828188  0.856350
                    Wuthering Heights  0.815333   0.799738 0.831293  0.815210
                         Frankenstein  0.843328   0.849145 0.829787  0.839354
The Adventures of Alice in Wonderland  0.883562   0.885655 0.900634  0.893082

## Key Insights

1. **Balanced Training**: Each model was trained on a balanced dataset with equal numbers of positive and negative samples to avoid class imbalance issues.

2. **Binary Classification**: Each model performs binary classification (belongs to book or not) rather than multi-label classification.

3. **Performance Range**: F1 scores range from 0.8152 to 0.8931, showing consistent performance across all books.

4. **Model Consistency**: All models achieve accuracy above 80%, indicating good discriminative ability.

## Comparison with Previous Approaches

This approach differs from the previous multi-label classification approach by:
- Training separate models for each book
- Using binary classification instead of multi-label
- Creating balanced datasets for each book
- Potentially better handling of class imbalance

## Files Generated
- Model files: `models/*_best_model.pth`
- Results: `models/individual_book_results.json`
- Visualizations: `models/individual_book_model_comparison.png`, `models/individual_book_analysis.png`
