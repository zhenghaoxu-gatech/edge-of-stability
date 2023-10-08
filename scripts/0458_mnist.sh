#!/bin/bash

# for weight in "w1_10_w2_40"
# do

#     for lr in 0.04 0.05 0.07 0.09 0.12
#     do
#         python src/gd.py mnist fc-relu-depth1  mse $lr 10000 --batch_norm --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
#     done
    
# done

for weight in "w1_6_w2_15"
do
    for lr in 0.00001 0.00002 0.00003 0.00004 0.00005
    do
        python src/gd.py mnist fc-cubic_relu-outer  huber $lr 10000 --neigs 1  --eig_freq 400 --width 200 --init_weight $weight
    done

    
done