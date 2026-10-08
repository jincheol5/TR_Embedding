#!/bin/bash
set -e

python -m app.train_model --model_name TGAT --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name DyGFormer --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1

python -m app.train_model --model_name TGAT --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name DyGFormer --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --save_model 1
python -m app.train_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --save_model 1