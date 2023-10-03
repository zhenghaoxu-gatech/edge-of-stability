#!/bin/bash

for weight in "w1_6_w2_20"
do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
        python src/gd.py cifar10-1k fc-tanh-depth1  huber $lr 100000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done
done