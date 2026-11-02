#!/bin/bash
set -e

DATASETS=(CollegeMsg soc-sign-bitcoinalpha soc-sign-bitcoinotc)

SEEDS=(1 2 3)

for dataset_name in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python -m app.SNAP.save_sample_loader --dataset_name "$dataset_name" --purpose val --seed "$seed" --batch_size 200
        python -m app.SNAP.save_sample_loader --dataset_name "$dataset_name" --purpose test --seed "$seed" --batch_size 200
    done
done
