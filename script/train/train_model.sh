#!/bin/bash

DATASETS=(CollegeMsg soc-sign-bitcoinalpha soc-sign-bitcoinotc)
SEEDS=(1 2 3)
LRS=(0.0001 0.0005)
MODELS=(TGAT TGN CAW-N GraphMixer DyGFormer TPNet ReaCH-TGN)

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for lr in "${LRS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "Training: dataset=$dataset, seed=$seed, lr=$lr, model=$model"

                if [ "$model" == "GraphMixer" ]; then
                    python -m app.train.train_model --model_name "$model" --seed "$seed" --lr "$lr" --one_hot 1 --dataset_name "$dataset" --save_model 1
                else
                    python -m app.train.train_model --model_name "$model" --seed "$seed" --lr "$lr" --dataset_name "$dataset" --save_model 1
                fi
            done
        done
    done
done