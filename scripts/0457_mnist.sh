#!/bin/bash

for weight in "w1_10_w2_40"
do
    for lr in 0.0002 0.0003 0.0004 0.0005
    do
        python src/gd.py mnist fc-relu-depth1  mse $lr 100000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

    for lr in 0.0001 0.00012 0.00014 0.00016 0.00018
    do
        python src/gd.py mnist fc-cubic_relu-depth1  mse $lr 100000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

    for lr in 0.001 0.003 0.005 0.007 0.009
    do
        python src/gd.py mnist fc-tanh-depth1  mse $lr 100000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
        python src/gd.py mnist fc-relu-depth1  huber $lr 100000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
    
done