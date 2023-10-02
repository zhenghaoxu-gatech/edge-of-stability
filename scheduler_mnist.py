import os
import numpy as np


lr_list_mse_relu = {
    "w_init": [0.03, 0.05, 0.08],
    "w1_10_w2_2": [0.03, 0.05, 0.1], 

    "w1_10_w2_5": [0.01, 0.02, 0.03], 
    "w1_20_w2_5": [0.01, 0.02, 0.03], 
    "w1_20_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_20": [0.01, 0.02, 0.03], 

    "w1_1_w2_10": [0.0008, 0.001, 0.0012, 0.0014],

    }
lr_list_mse_tanh = {
    "w_init": [0.01, 0.02, 0.03], 
    "w1_10_w2_2": [0.01, 0.02, 0.03], 

    "w1_10_w2_5": [0.005, 0.01, 0.015], 
    "w1_20_w2_5": [0.005, 0.01, 0.015], 
    "w1_20_w2_10": [0.005, 0.01, 0.015], 
    "w1_10_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_20": [0.01, 0.02, 0.03],
    
    "w1_1_w2_10": [0.0001, 0.0005, 0.0009],
}

lr_list_huber_relu = {
    "w_init": [0.03, 0.05, 0.08], 
    "w1_10_w2_2": [0.03, 0.05, 0.1], 

    "w1_10_w2_5": [0.03, 0.05, 0.08], 
    "w1_20_w2_5": [0.03, 0.05, 0.08], 
    "w1_20_w2_10": [0.03, 0.05, 0.08], 
    "w1_10_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_20": [0.01, 0.02, 0.03],
    
    "w1_1_w2_10": [0.0005, 0.001, 0.0015],
    }
lr_list_huber_tanh = {
    "w_init": [0.01, 0.03, 0.05], 
    "w1_10_w2_2": [0.01, 0.03, 0.05], 

    "w1_10_w2_5": [0.01, 0.02, 0.03], 
    "w1_20_w2_5": [0.01, 0.02, 0.03], 
    "w1_20_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_10": [0.01, 0.02, 0.03], 
    "w1_10_w2_20": [0.01, 0.02, 0.03],

    "w1_1_w2_10": [0.0002, 0.0008, 0.0014],
    
}

lr_list_mse_relu_BN = {
    "w_init": [0.05, 0.06, 0.07], 
    "w1_10_w2_2": [0.05, 0.06, 0.07], 

    }

lr_list_mse_cubicrelu = {
    "w_init": [0.0004, 0.0006, 0.0008],
    "w1_10_w2_2": [0.0004, 0.0005, 0.0006, 0.0007, 0.0008],

    }

# for weight in ["w1_10_w2_5", "w10_10_w2_10", "w1_20_w2_5", "w1_20_w2_10"]:
# for weight in ["w1_20_w2_10", "w1_10_w2_10", "w1_10_w2_20"]:
for weight in ["w1_1_w2_10"]:
    for lr in lr_list_mse_relu[weight]:
        os.system(f"python src/gd.py mnist fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}") 
    for lr in lr_list_mse_tanh[weight]:
        os.system(f"python src/gd.py mnist fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}") 
    for lr in lr_list_huber_relu[weight]:
        os.system(f"python src/gd.py mnist fc-relu-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}") 
    for lr in lr_list_huber_tanh[weight]:
        os.system(f"python src/gd.py mnist fc-tanh-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    # for lr in lr_list_mse_relu_BN[weight]:
        # os.system(f"python src/gd.py mnist fc-relu-depth1  mse {lr} 20000 --neigs 2 --batch_norm  --eig_freq 40 --width 200 --init_weight {weight}")  