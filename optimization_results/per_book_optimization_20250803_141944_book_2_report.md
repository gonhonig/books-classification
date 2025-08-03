# Hyperparameter Optimization Report

**Model Type**: per_book
**Book**: Wuthering Heights
**Study Name**: per_book_optimization_20250803_141944_book_2
**Best Trial Value**: 0.8538
**Number of Trials**: 8

## Best Parameters
```json
{
  "n_layers": 9,
  "hidden_dim_0": 224,
  "hidden_dim_1": 129,
  "hidden_dim_2": 93,
  "hidden_dim_3": 131,
  "hidden_dim_4": 299,
  "hidden_dim_5": 415,
  "hidden_dim_6": 465,
  "hidden_dim_7": 287,
  "hidden_dim_8": 417,
  "learning_rate": 0.0003330327583358534,
  "batch_size": 32,
  "dropout_rate": 0.4445833457366378,
  "weight_decay": 1.0965600286167261e-05,
  "epochs": 83,
  "patience": 28
}
```

## Parameter Importance
- **hidden_dim_0**: 0.1883
- **hidden_dim_2**: 0.1838
- **hidden_dim_1**: 0.1286
- **n_layers**: 0.0775
- **patience**: 0.0634
- **hidden_dim_3**: 0.0606
- **learning_rate**: 0.0580
- **dropout_rate**: 0.0576
- **hidden_dim_7**: 0.0483
- **hidden_dim_6**: 0.0469
- **weight_decay**: 0.0323
- **epochs**: 0.0311
- **hidden_dim_4**: 0.0171
- **hidden_dim_5**: 0.0064
- **batch_size**: 0.0001

## Trial History
| Trial | Value | Status |
|-------|-------|--------|
| 0 | 0.8488 | COMPLETE |
| 1 | 0.8355 | COMPLETE |
| 2 | 0.8474 | COMPLETE |
| 3 | 0.1422 | COMPLETE |
| 4 | 0.8476 | COMPLETE |
| 5 | 0.8498 | COMPLETE |
| 6 | 0.8411 | COMPLETE |
| 7 | 0.8538 | COMPLETE |
