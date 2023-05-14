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
for lr in np.array([0.002, 0.003, 0.004]):
    os.system(f"python src/gd.py cifar10-1k fc-relu-depth1  mse {lr} 5000 --neigs 2  --eig_freq 10")
    os.system(f"python src/gd.py cifar10-1k fc-tanh-depth1  mse {lr} 5000 --neigs 2  --eig_freq 10")
    os.system(f"python src/gd.py cifar10-1k fc-leaky_relu-depth1  mse {lr} 5000 --neigs 2  --eig_freq 10")
