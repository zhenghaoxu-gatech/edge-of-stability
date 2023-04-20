import os
import numpy as np

np.random.bit_generator = np.random._bit_generator

for lr in np.arange(0.1, 0.2, 0.02):
    os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
    os.system(f"python src/gd.py cifar10-5k fc-relu-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
for lr in np.arange(0.01, 0.1, 0.02):
    os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  mse {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")
    os.system(f"python src/gd.py cifar10-5k fc-tanh-depth1  ce {lr} 5000 --acc_goal 0.99 --neigs 2  --eig_freq 10")