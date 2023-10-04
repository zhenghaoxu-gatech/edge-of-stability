#!/bin/bash

for weight in "w1_6_w2_20"
do
    for lr in 0.00002 0.00006 0.0001 0.00014 0.00018
    do
        python src/gd.py cifar10-1k fc-cubic_relu-depth1  mse $lr 100000 --batch_norm --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
done