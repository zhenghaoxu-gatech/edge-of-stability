import os
import numpy as np


lr_list_mse_relu_mnist = {
    "w1_10_w2_40": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
    "w1_12_w2_40": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
    }
lr_list_mse_tanh_mnist = {
    "w1_10_w2_40": [0.001, 0.003, 0.005, 0.007, 0.009],
    # "w1_10_w2_40": [0.003, 0.007, 0.009],
}

lr_list_huber_relu_mnist = {
    "w1_10_w2_40": [0.001, 0.003, 0.005, 0.007, 0.009],
    # "w1_10_w2_40": [0.003, 0.007, 0.009],
    }
lr_list_huber_tanh_mnist = {
    "w1_10_w2_40": [0.001, 0.003, 0.005, 0.007, 0.009],
    # "w1_10_w2_40": [0.003, 0.007, 0.009],
}

lr_list_mse_relu_BN_mnist = {
    "w1_10_w2_40": [0.007, 0.01, 0.015, 0.02, 0.025, 0.03],
    "w1_12_w2_40": [0.05, 0.1, 0.15],
    "w1_6_w2_15": [0.01, 0.03, 0.05, 0.07, 0.09],
    }

lr_list_mse_cubicrelu_mnist = {
    # "w1_10_w2_40": [0.0001, 0.0005],
    "w1_10_w2_40": [0.0001, 0.00012, 0.00014, 0.00016, 0.00018],
    # "w1_6_w2_20": [0.00002, 0.00006, 0.0001, 0.00014, 0.00018],
    }
lr_list_mse_cubicrelu_BN_mnist = {
    # "w1_10_w2_40": [0.0001, 0.0005],
    "w1_10_w2_40": [0.002, 0.005, 0.01],
    # "w1_6_w2_20": [0.00002, 0.00006, 0.0001, 0.00014, 0.00018],
    }
lr_list_huber_cubicrelu_outer_mnist = {
    "w1_10_w2_40": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
    }


for weight in ["w1_10_w2_40"]:
    # for lr in lr_list_mse_relu_mnist[weight]: # must over 2/eta
    #     os.system(f"python src/gd.py mnist fc-relu-depth1  mse {lr} 800 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    # for lr in lr_list_mse_cubicrelu_BN_mnist[weight]:
    for lr in [0.01]:
        os.system(f"python src/gd.py mnist fc-cubic_relu-depth1  mse {lr} 4000 --batch_norm --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
#     for lr in lr_list_mse_cubicrelu_mnist[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-depth1  mse {lr} 200000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
#     for lr in lr_list_mse_tanh_mnist[weight]:
#         os.system(f"python src/gd.py mnist fc-tanh-depth1  mse {lr} 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight {weight}")

#     for lr in lr_list_huber_relu_mnist[weight]:
#         os.system(f"python src/gd.py mnist fc-relu-depth1  huber {lr} 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight {weight}")

#     ############################

#     for lr in lr_list_huber_tanh_mnist[weight]:
#         os.system(f"python src/gd.py mnist fc-tanh-depth1  huber {lr} 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight {weight}")


# for weight in ["w1_8_w2_20"]:
#     for lr in lr_list_huber_cubicrelu_outer_mnist[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-outer  huber {lr} 200000 --neigs 1  --eig_freq 400 --width 200 --init_weight {weight}")  
#     for lr in lr_list_mse_cubicrelu_outer[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-outer  mse {lr} 400 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

#     for lr in lr_list_huber_cubicrelu_outer[weight]:
#         os.system(f"python src/gd.py mnist fc-cubic_relu-outer  huber {lr} 400 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
#         break
