#!/bin/bash
set -e

python -m app.train.train_model --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg

python -m app.train.train_model --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg

python -m app.train.train_model --model_name CAW-N --seed 1 --lr 0.0001 --dataset_name CollegeMsg

python -m app.train.train_model --model_name GraphMixer --seed 1 --lr 0.0001 --one_hot 1 --dataset_name CollegeMsg

python -m app.train.train_model --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg

python -m app.train.train_model --model_name TPNet --seed 1 --lr 0.0001 --dataset_name CollegeMsg

python -m app.train.train_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg