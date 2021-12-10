#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate deep_learning
python bias_correction.py --input_dir /itet-stor/zhexwu/net_scratch/data/MnMs --output_dir /itet-stor/zhexwu/net_scratch/data/MnMs_extracted

