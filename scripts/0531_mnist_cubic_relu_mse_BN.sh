#!/bin/bash

for weight in "w1_10_w2_40"
do
    for lr in 0.001 0.002 0.003 0.004 0.005
    do
        python src/gd.py mnist fc-cubic_relu-depth1  huber $lr 1000000 --batch_norm --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
done
