#!/bin/bash

for weight in "w1_10_w2_40"
do
    for lr in 0.001 0.003 0.005 0.007 0.009
    do
        python src/gd.py mnist fc-tanh-depth1  huber $lr 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

    for lr in 0.007 0.01 0.015 0.02 0.025 0.03
    do
        python src/gd.py mnist fc-relu-depth1  mse $lr 200000 --batch_norm --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
    
done

for weight in "w1_8_w2_20"
do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
        python src/gd.py mnist fc-cubic_relu-outer  huber $lr 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

    
done