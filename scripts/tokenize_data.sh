#!/bin/bash

curr_path=$(pwd)
echo $curr_path
python flow_matching/data_utils.py --config_name bert-base-uncased --data_dir $curr_path/datasets/QQP/ --seq_len 64 --split train
python flow_matching/data_utils.py --config_name bert-base-uncased --data_dir $curr_path/datasets/QQP/ --seq_len 64 --split valid
python flow_matching/data_utils.py --config_name bert-base-uncased --data_dir $curr_path/datasets/QQP/ --seq_len 64 --split test