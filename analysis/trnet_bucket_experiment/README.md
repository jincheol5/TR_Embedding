# TRNet bucket experiment

CollegeMsg: replay 41,858 training events and 8,970 validation events; evaluate 9,000 saved validation pairs (4,500 per class). Three random seeds. No decoder training or model source edits.

Each bucket has a direction and log norm. Budget = K*(d+1) float32 values per node, plus an equally sized timestamp snapshot. Immutable initial vectors, hash indices, and timestamps are excluded. Source hashes are fixed, independent of labels. Same-time propagation reads pre-update source state across batch boundaries.

| State floats/node | Buckets | Dimensions/bucket | Cosine AUC mean ± seed std | Full-dot AUC mean ± seed std | Cosine AP mean |
|---:|---:|---:|---:|---:|---:|
| 33 | 1 | 32 | 0.5005 ± 0.0026 | 0.4937 ± 0.0152 | 0.5068 |
| 64 | 1 | 63 | 0.5021 ± 0.0193 | 0.4995 ± 0.0171 | 0.5067 |
| 64 | 4 | 15 | 0.5132 ± 0.0282 | 0.5131 ± 0.0269 | 0.5168 |
| 64 | 16 | 3 | 0.5009 ± 0.0115 | 0.4969 ± 0.0074 | 0.5046 |
| 256 | 1 | 255 | 0.5014 ± 0.0102 | 0.4991 ± 0.0072 | 0.5173 |
| 256 | 4 | 63 | 0.5149 ± 0.0034 | 0.5056 ± 0.0050 | 0.5231 |
| 256 | 16 | 15 | 0.5107 ± 0.0160 | 0.5038 ± 0.0157 | 0.5258 |
| 1024 | 1 | 1023 | 0.5182 ± 0.0121 | 0.5154 ± 0.0107 | 0.5165 |
| 1024 | 4 | 255 | 0.5181 ± 0.0101 | 0.5165 ± 0.0128 | 0.5284 |
| 1024 | 16 | 63 | 0.5115 ± 0.0145 | 0.5067 ± 0.0148 | 0.5277 |

Full dot uses the monotonic transform sign(x)*log(1+abs(x)); no large real vector is reconstructed.

Bucket partitioning did not produce consistently useful separation in these tested settings. Increasing dimension alone also remained near chance. This is a representation-level screening experiment, not a trained decoder benchmark. Three seeds do not establish statistical significance, and other hash counts / query decoders are untested.

The independent NumPy replay engine was checked against direct sums with same-time events and against the repository single-bucket implementation. The 32-dimensional reference also reproduces the earlier near-chance findings.

Run from repository root: `/opt/anaconda3/envs/tgn/bin/python analysis/trnet_bucket_experiment/run_experiment.py`
