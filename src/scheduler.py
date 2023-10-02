import os
import numpy as np

'''
20230422
'''
# for lr in np.arange(0.01, 0.1, 0.02):
#     os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
# for lr in np.arange(0.01, 0.1, 0.02):
#     os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
# for lr in np.arange(0.01, 0.1, 0.02):
#     os.system(f"python src/gd.py cifar10-5k fc-leaky_relu-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-leaky_relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
# for lr in np.arange(0.01, 0.1, 0.02):
#     os.system(f"python src/gd.py cifar10-5k fc-elu-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-elu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")

'''
20230423
''' 
# for lr in np.arange(0.18, 0.3, 0.02):
#     os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-leaky_relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-5k fc-elu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")

'''
20230501
- bias=False, check archs.py
- |w1|=1.0, |w2|=10.0
- width = 30
''' 
# for lr in np.arange(0.1, 0.25, 0.01):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")
#     os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")
#     os.system(f"python src/gd.py cifar10-1k fc-elu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")

# for lr in np.arange(4e-4, 5e-4, 2e-5):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")


# |w2|=100
# for lr in np.arange(0.2, 0.5, 0.01):
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 5")

'''
20230505
- bias=False, check archs.py
- |w1|=1.0, |w2|=10.0
- width = 200
- run longer
''' 
# for lr in np.arange(0.1, 0.25, 0.03):
#     os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  ce {lr} 5000 --neigs 2  --eig_freq 20")
#     os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  ce {lr} 5000 --neigs 2  --eig_freq 20")

'''
20230505
- bias=True, check archs.py
- |w1|=1.0, |w2|=10.0
- width = 30
- run longer
''' 
# for lr in np.arange(0.1, 0.25, 0.03):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  ce {lr} 2000 --neigs 2  --eig_freq 10")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  ce {lr} 2000 --neigs 2  --eig_freq 10")

# for lr in np.arange(0.001, 0.0012, 0.0003):
    # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  ce {lr} 2000 --neigs 2  --eig_freq 10")
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  ce {lr} 5000 --neigs 2  --eig_freq 10")
    # os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  ce {lr} 2000 --neigs 2  --eig_freq 10")

'''
- bias=True
- |w_1|=5.0, |w_2|=5.0
- width=300
- run longer
'''
# for lr in np.arange(0.013, 0.025, 0.003):
# for lr in np.array([0.001]):
    # os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  ce {lr} 5000 --neigs 2  --eig_freq 10")
    # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 3000 --neigs 2  --eig_freq 10")
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 3000 --neigs 2  --eig_freq 10")

'''
20230513
- bias=True, check archs.py
- |w1|=1.0, |w2|=10.0
- width = 30
- run longer
- init_sharp=4200 -> eta=0.0004
''' 
# for lr in np.array([0.001]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 3000 --neigs 2  --eig_freq 10")

'''
- bias=False
'''
# for lr in np.array([0.001]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 3000 --neigs 2  --eig_freq 10")
# for lr in np.array([0.0012]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 3000 --neigs 2  --eig_freq 10")


'''
- |w1|=2.0, |w2|=10.0
- bias=False / True (unchanged) / True (b1=b2=2.0)
'''

'''
- |w1|=2.0->1.0
- relu->tanh
'''

'''
- |w1|=1.0
- tanh->leaky_relu alpha=1/5.5 (https://arxiv.org/pdf/1505.00853.pdf)
- init_sharp=3700
- 0.001: diverge
'''

'''
- bias=init
- |w1|=1, |w2|=5
- freq=20, steps=10000
'''
# for lr in np.array([0.0005, 0.0008, 0.001]):
    # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 10000 --neigs 1  --eig_freq 20 --width 200 --bias --init_bias b_init --init_weight w1_1_w2_10")
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 10000 --neigs 1  --eig_freq 20 --width 200 --bias --init_bias b_init --init_weight w1_1_w2_10")
    # os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  mse {lr} 10000 --neigs 1  --eig_freq 20 --width 200 --bias --init_bias b_init --init_weight w1_1_w2_10")


'''
20230516
- huber loss -> softmax
- bias=False
- |w1|=1, |w2|=10
- freq=20, steps=10000
'''
# for lr in np.array([0.01, 0.02, 0.04]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")
#     os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")

'''
20230602
- ReLU+BN -> EOS
- bias=False
- |w1|=1, |w2|=10
- freq=20, steps=10000
'''
# for lr in np.array([0.001, 0.002, 0.004, 0.01, 0.02]):
    # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")
    # os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_10")


'''
20230613
- bias=False
- |w1|=1, |w2|=1
- freq=20, steps=10000
'''
# for lr in np.array([0.001, 0.002, 0.004, 0.01, 0.02]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_1")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_1")

# for lr in np.array([0.001, 0.002]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_2")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_2")

# for lr in np.array([0.004, 0.01, 0.02]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_2")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_2")

# for lr in np.array([0.001, 0.002, 0.004, 0.01, 0.02]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_4")
#     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_4")


# for lr in np.array([0.0001, 0.0002, 0.0004, 0.001]):
#     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_8")
# for lr in np.array([0.04, 0.1, 0.2]):
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_2")
    # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight w1_1_w2_4")

'''
20230623
- bias=False
- |w1|=1, |w2|=1
'''

lr_list_mse_relu = {
    "w1_6_w2_20": [0.00021, 0.00023, 0.00025, 0.00027],
    }
lr_list_mse_tanh = {
    "w1_6_w2_20": [0.00035, 0.0004, 0.00045],
}

lr_list_huber_relu = {
    "w1_6_w2_20": [0.0003, 0.0005, 0.0007, 0.0009],
    }
lr_list_huber_tanh = {
    "w1_6_w2_20": [0.00015, 0.0002, 0.00025],
}

lr_list_mse_relu_BN = {
    "w1_6_w2_20": [0.0001, 0.00015, 0.0002, 0.00025, 0.0003],
    }

lr_list_mse_cubicrelu = {
    "w1_6_w2_20": [0.00002, 0.00006, 0.0001, 0.00014, 0.00018],
    }

lr_list_huber_cubicrelu = {
    "w1_6_w2_20": [0.0001, 0.0003, 0.0005, 0.0007, 0.0009],
    }

# lr_list_mse_cubicrelu_double = {
#     "w1_6_w2_15": [0.000005, 0.00001, 0.000015],
#     }
# lr_list_huber_cubicrelu_double = {
#     "w1_6_w2_15": [0.00006, 0.00009],
#     }
lr_list_mse_cubicrelu_outer = {
    "w1_6_w2_20": [0.001, 0.002, 0.003],
    }
lr_list_huber_cubicrelu_outer = {
    "w1_6_w2_20": [0.001, 0.002, 0.005],
    }



# for weight in ["w1_6_w2_20"]:
    # for lr in lr_list_mse_relu[weight]: # must over 2/eta
        # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 5000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")
    
    # for lr in lr_list_mse_cubicrelu[weight]:
        # os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-depth1  mse {lr} 100000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

    # for lr in lr_list_huber_cubicrelu[weight]:
    #     os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
    # for lr in lr_list_mse_tanh[weight]:
        # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    # for lr in lr_list_huber_relu[weight]:
        # os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    # for lr in lr_list_huber_tanh[weight]:
        # os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  huber {lr} 5000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

    # for lr in lr_list_mse_relu_BN[weight]:
    #     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --batch_norm --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

for weight in ["w1_6_w2_20"]:
    # for lr in [0.000005]:
    # for lr in lr_list_mse_cubicrelu_outer[weight]:
        # os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-outer  mse {lr} 5000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

    # for lr in lr_list_huber_cubicrelu_outer[weight]:
    for lr in [0.001]:
        os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-outer  huber {lr} 5000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

'''
Extension
'''
# for weight in ["w1_6_w2_20"]:
#     for lr in lr_list_mse_relu[weight]: # must over 2/eta
#         os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")
    
#     # for lr in lr_list_mse_cubicrelu[weight]:
#         # os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-depth1  mse {lr} 100000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

#     for lr in lr_list_huber_cubicrelu[weight]:
#         os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  

#     for lr in lr_list_huber_cubicrelu_double[weight]:
#         os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-double  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
#     # for lr in lr_list_huber_cubicrelu_outer[weight]:
#         # os.system(f"python src/gd.py cifar10-1k fc-cubic_relu-outer  huber {lr} 400 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    

#     # for lr in lr_list_mse_tanh[weight]:
#     #     os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

#     # for lr in lr_list_huber_relu[weight]:
#     #     os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

#     for lr in lr_list_huber_tanh[weight]:
#         os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  huber {lr} 20000 --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")

#     for lr in lr_list_mse_relu_BN[weight]:
#         os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 20000 --batch_norm --neigs 1  --eig_freq 40 --width 200 --init_weight {weight}")  
    
        

'''
initial sharpness for w2=1, 2, 5, 10
MSE+ReLU: 28.2788, 112.0073, 699.3280, 2797.0710
MSE+Tanh: 102.5512, 408.8297, 2552.8914, 10209.6406
Huber+ReLU: 27.0230, 106.8618, 667.1134, 2668.2041
Huber+Tanh: 97.3116, 387.8902, 2422.0889, 9680.3252
CE+ReLU: 3.9599, 11.4496, 68.9133, 275.4725

MSE+ReLU+BN: 

step size:
MSE+ReLU: 
    (0.0707) 0.08, 0.09 0.1, 0.11 (0.12)
    (0.0178) 0.02, 0.025, 0.03 (0.03)
    (0.0028) 0.003, 0.0035, 0.004
    (0.0007) 0.0008, 0.001, 0.0012
'''

'''
0717
initialization: no balancing: far from minimizer.
finer step size
mnist: 28*28
'''