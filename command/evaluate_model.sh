#!/bin/bash
set -e

python -m app.evaluate_model --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type random
python -m app.evaluate_model --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type random
python -m app.evaluate_model --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type random
python -m app.evaluate_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type random

python -m app.evaluate_model --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type focused
python -m app.evaluate_model --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type focused
python -m app.evaluate_model --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type focused
python -m app.evaluate_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type focused

python -m app.evaluate_model --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type hop_range
python -m app.evaluate_model --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type hop_range
python -m app.evaluate_model --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type hop_range
python -m app.evaluate_model --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --evaluate_type hop_range