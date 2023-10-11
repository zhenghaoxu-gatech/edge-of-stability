#!/bin/bash

for weight in "w1_10_w2_40"
do
    for lr in 0.001 0.003 0.005 0.007 0.009
    do
        python src/gd.py mnist fc-relu-depth1  huber $lr 500000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
done
