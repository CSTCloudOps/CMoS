#!/bin/bash

for pred_len in 96 192 336 720
do
    echo "Running with pred_len=${pred_len}"
    CUDA_VISIBLE_DEVICES=0 python Pipeline/OneMulEdit.py --pred_len $pred_len --dataset weather
done