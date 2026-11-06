#!/bin/bash
set -e

# CollegeMsg
python -m app.evaluate.evaluate_model --model_name TGAT --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name TGN --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name CAW-N --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name GraphMixer --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name DyGFormer --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name TPNet --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name ReaCH-TGN --dataset_name CollegeMsg
python -m app.evaluate.evaluate_model --model_name TRNet --dataset_name CollegeMsg


# soc-sign-bitcoin-alpha
python -m app.evaluate.evaluate_model --model_name TGAT --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name TGN --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name CAW-N --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name GraphMixer --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name DyGFormer --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name TPNet --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name ReaCH-TGN --dataset_name soc-sign-bitcoinalpha 
python -m app.evaluate.evaluate_model --model_name TRNet --dataset_name soc-sign-bitcoinalpha 


# soc-sign-bitcoin-otc
python -m app.evaluate.evaluate_model --model_name TGAT --dataset_name soc-sign-bitcoinotc  
python -m app.evaluate.evaluate_model --model_name TGN --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name CAW-N --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name GraphMixer --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name DyGFormer --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name TPNet --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name ReaCH-TGN --dataset_name soc-sign-bitcoinotc
python -m app.evaluate.evaluate_model --model_name TRNet --dataset_name soc-sign-bitcoinotc