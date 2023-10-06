import os
import numpy as np


lr_list_mse_relu_mnist = {
    "w1_6_w2_20": [0.01, 0.02, 0.03],
    "w1_4_w2_20": [0.02, 0.03],
    "w1_6_w2_15": [0.02, 0.03],
    "w1_3_w2_10": [0.02, 0.04, 0.06],
    "w1_2_w2_10": [0.02, 0.05, 0.09],
    "w1_4_w2_18": [0.02, 0.05, 0.09],
    "w1_5_w2_25": [0.004, 0.005, 0.006],
    "w1_8_w2_20": [0.005, 0.006],
    "w1_15_w2_40": [0.005, 0.006],
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    }
lr_list_mse_tanh_mnist = {
    "w1_6_w2_20": [0.0003, 0.00035, 0.0004, 0.00045],
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
}

lr_list_huber_relu_mnist = {
    "w1_6_w2_20": [0.0003, 0.0005, 0.0007, 0.0009],
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    }
lr_list_huber_tanh_mnist = {
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    "w1_6_w2_20": [0.0003, 0.0004, 0.0005],
}

lr_list_mse_relu_BN_mnist = {
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    "w1_6_w2_20": [0.0001, 0.00015, 0.0002, 0.00025, 0.0003],
    }

lr_list_mse_cubicrelu_mnist = {
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    "w1_6_w2_20": [0.00002, 0.00006, 0.0001, 0.00014, 0.00018],
    }
lr_list_huber_cubicrelu_mnist = {
    "w1_10_w2_40": [0.0001, 0.0005, 0.001, 0.005],
    "w1_6_w2_20": [0.00002, 0.00006, 0.0001, 0.00014, 0.00018],
    }


for weight in ["w1_10_w2_40"]:
    for lr in lr_list_mse_relu_mnist[weight]: # must over 2/eta
        os.system(f"python src/gd.py mnist fc-relu-depth1  mse {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")
    
    for lr in lr_list_mse_cubicrelu_mnist[weight]:
        os.system(f"python src/gd.py mnist fc-cubic_relu-depth1  mse {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

    for lr in lr_list_huber_cubicrelu_mnist[weight]:
        os.system(f"python src/gd.py mnist fc-cubic_relu-depth1  huber {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
    for lr in lr_list_mse_tanh_mnist[weight]:
        os.system(f"python src/gd.py mnist fc-tanh-depth1  mse {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    for lr in lr_list_huber_relu_mnist[weight]:
        os.system(f"python src/gd.py mnist fc-relu-depth1  huber {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    for lr in lr_list_huber_tanh_mnist[weight]:
        os.system(f"python src/gd.py mnist fc-tanh-depth1  huber {lr} 4000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

#     for lr in lr_list_mse_relu_BN[weight]:
#         os.system(f"python src/gd.py mnist fc-relu-depth1  mse {lr} 400 --batch_norm --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

# for weight in ["w1_3_w2_10"]:
#     for lr in lr_list_mse_cubicrelu_outer[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-outer  mse {lr} 400 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

#     for lr in lr_list_huber_cubicrelu_outer[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-outer  huber {lr} 400 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
#         break
