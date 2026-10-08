# Temporal Graph Embedding for Temporal Reachability Prediction
### Implemented models:
> TGAT
> 
> TGN
> 
> DyGFormer
> 
> ReaCH-TGN

### Evaluation
> - Trained using Random Sampling
> - Dataset: CollegeMsg
> - Seed: 1
> - Learning Rate: 0.0001
> - Batch Size: 200

| Model | Random | Dst Focused | Hop Range (1 <= Hop < 3) | Hop Range (3 <= Hop < 5) | Hop Range (5 <= Hop) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TGAT | 70.91 | 90.95 | 55.11 | 52.88 | 34.23 |
| TGN | 80.79 | 87.03 | 84.30 | 75.38 | 60.46 |
| DyGFormer | 60.56 | 84.42 | 43.69 | 28.73 | 14.92 |
| ReaCH-TGN | 63.38 | 70.50 | 87.30 | 79.00 | 72.11 |
