#!/usr/bin/env bash

python ./run_alternating_opt.py --dataset_name "csf" --dataset_root "./data/MnMs" --device "cpu" --batch_size 4 --model_save_dir "./params/norm_u_net" --eval_plot_save_dir "./images/norm_u_net"

