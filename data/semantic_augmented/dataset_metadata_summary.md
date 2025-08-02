# Semantic Augmented Dataset - Comprehensive Statistics

## Dataset Overview
- **Total Samples**: 10120
- **Number of Classes**: 4
- **Dataset Type**: semantic_augmented_balanced
- **Multi-label**: True

## Split Distribution
- **Train**: 7083 samples
- **Validation**: 1518 samples
- **Test**: 1519 samples

## Overall Label Distribution
- **1_label**: 4966 samples
- **2_labels**: 3479 samples
- **3_labels**: 1264 samples
- **4_labels**: 411 samples

## Per-Split Statistics
### Train Split
- **Total Samples**: 7083
- **Label Distribution**:
  - 1_label: 3443 samples
  - 2_labels: 2429 samples
  - 3_labels: 911 samples
  - 4_labels: 300 samples
- **Book Distribution**:
  - Anna Karenina: 3511 positive, 3572 negative
  - Frankenstein: 3104 positive, 3979 negative
  - The Adventures of Alice in Wonderland: 2081 positive, 5002 negative
  - Wuthering Heights: 3538 positive, 3545 negative

### Validation Split
- **Total Samples**: 1518
- **Label Distribution**:
  - 1_label: 776 samples
  - 2_labels: 524 samples
  - 3_labels: 171 samples
  - 4_labels: 47 samples
- **Book Distribution**:
  - Anna Karenina: 726 positive, 792 negative
  - Frankenstein: 658 positive, 860 negative
  - The Adventures of Alice in Wonderland: 426 positive, 1092 negative
  - Wuthering Heights: 715 positive, 803 negative

### Test Split
- **Total Samples**: 1519
- **Label Distribution**:
  - 1_label: 747 samples
  - 2_labels: 526 samples
  - 3_labels: 182 samples
  - 4_labels: 64 samples
- **Book Distribution**:
  - Anna Karenina: 763 positive, 756 negative
  - Frankenstein: 681 positive, 838 negative
  - The Adventures of Alice in Wonderland: 410 positive, 1109 negative
  - Wuthering Heights: 747 positive, 772 negative

## Balanced Dataset Configuration
- **target_samples_per_class**: 5000
- **multi_label_priority**: True
- **balanced_positive_negative**: True
- **total_samples**: 10120
- **multi_label_samples**: 5154
- **single_label_samples**: 4966
- **book_statistics**: {'Anna Karenina': {'positive_samples': 5000, 'negative_samples': 5120, 'total_samples': 10120}, 'Frankenstein': {'positive_samples': 4443, 'negative_samples': 5677, 'total_samples': 10120}, 'The Adventures of Alice in Wonderland': {'positive_samples': 2917, 'negative_samples': 7203, 'total_samples': 10120}, 'Wuthering Heights': {'positive_samples': 5000, 'negative_samples': 5120, 'total_samples': 10120}}
