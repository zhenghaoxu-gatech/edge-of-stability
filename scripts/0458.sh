#!/bin/bash

for weight in "w1_3_w2_10"
do
    for lr in 0.000014 0.000012
    do
        python src/gd.py cifar10-1k fc-cubic_relu-outer  huber $lr 500000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
done