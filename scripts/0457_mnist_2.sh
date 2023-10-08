#!/bin/bash

for weight in "w1_10_w2_40"
do

    for lr in 0.00012 
    do
        python src/gd.py mnist fc-cubic_relu-depth1  mse $lr 1000000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

  
done