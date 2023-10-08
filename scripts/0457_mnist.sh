#!/bin/bash

for weight in "w1_10_w2_40"
do
    for lr in 0.0003 0.0004 0.0005
    do
        python src/gd.py mnist fc-relu-depth1  mse $lr 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

  
done