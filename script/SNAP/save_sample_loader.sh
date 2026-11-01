#!/bin/bash
set -e

python -m app.SNAP.save_sample_loader --dataset_name CollegeMsg --purpose val --batch_size 200
python -m app.SNAP.save_sample_loader --dataset_name CollegeMsg --purpose test --batch_size 200

python -m app.SNAP.save_sample_loader --dataset_name soc-sign-bitcoinalpha --purpose val --batch_size 200
python -m app.SNAP.save_sample_loader --dataset_name soc-sign-bitcoinalpha --purpose test --batch_size 200

python -m app.SNAP.save_sample_loader --dataset_name soc-sign-bitcoinotc --purpose val --batch_size 200
python -m app.SNAP.save_sample_loader --dataset_name soc-sign-bitcoinotc --purpose test --batch_size 200