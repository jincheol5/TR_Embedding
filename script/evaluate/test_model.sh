#!/bin/bash
set -e

python -m app.evaluate.test_model --model_name TGAT --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TGAT --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TGAT --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  

python -m app.evaluate.test_model --model_name TGN --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TGN --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TGN --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  

python -m app.evaluate.test_model --model_name CAW-N --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name CAW-N --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name CAW-N --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  

python -m app.evaluate.test_model --model_name GraphMixer --dataset_name CollegeMsg --seed 1 --lr 0.0001 --one_hot 1
python -m app.evaluate.test_model --model_name GraphMixer --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 --one_hot 1
python -m app.evaluate.test_model --model_name GraphMixer --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001 --one_hot 1

python -m app.evaluate.test_model --model_name DyGFormer --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name DyGFormer --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name DyGFormer --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  

python -m app.evaluate.test_model --model_name TPNet --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TPNet --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name TPNet --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  

python -m app.evaluate.test_model --model_name ReaCH-TGN --dataset_name CollegeMsg --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name ReaCH-TGN  --dataset_name soc-sign-bitcoinalpha --seed 1 --lr 0.0001 
python -m app.evaluate.test_model --model_name ReaCH-TGN  --dataset_name soc-sign-bitcoinotc --seed 1 --lr 0.0001  