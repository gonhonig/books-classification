# Hyperparameter Optimization Report

**Model Type**: multi_label
**Study Name**: multi_label_optimization_20250803_134833
**Best Trial Value**: 0.8576
**Number of Trials**: 7

## Best Parameters
```json
{
  "n_layers": 8,
  "hidden_dim_0": 404,
  "hidden_dim_1": 430,
  "hidden_dim_2": 336,
  "hidden_dim_3": 251,
  "hidden_dim_4": 131,
  "hidden_dim_5": 330,
  "hidden_dim_6": 399,
  "hidden_dim_7": 227,
  "learning_rate": 0.00012367609745142668,
  "batch_size": 16,
  "dropout_rate": 0.45477314713190786,
  "weight_decay": 3.664611953105058e-06,
  "epochs": 109,
  "patience": 24
}
```

## Parameter Importance
- **hidden_dim_2**: 0.3306
- **hidden_dim_1**: 0.1367
- **dropout_rate**: 0.1137
- **hidden_dim_3**: 0.0902
- **hidden_dim_0**: 0.0617
- **n_layers**: 0.0598
- **learning_rate**: 0.0489
- **hidden_dim_5**: 0.0406
- **patience**: 0.0402
- **epochs**: 0.0334
- **weight_decay**: 0.0156
- **hidden_dim_4**: 0.0152
- **batch_size**: 0.0136

## Trial History
| Trial | Value | Status |
|-------|-------|--------|
| 0 | 0.8171 | COMPLETE |
| 1 | 0.8576 | COMPLETE |
| 2 | 0.8257 | COMPLETE |
| 3 | 0.8008 | COMPLETE |
| 4 | 0.8551 | COMPLETE |
| 5 | 0.8481 | COMPLETE |
| 6 | 0.8435 | COMPLETE |
