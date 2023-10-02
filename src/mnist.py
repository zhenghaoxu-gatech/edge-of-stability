import numpy as np
from torchvision.datasets import MNIST
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F

# DATASETS_FOLDER = os.environ["DATASETS"]
DATASETS_FOLDER = "./dataset"

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = 0.1307
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = 0.3081
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)
    elif loss == "huber":
        return _one_hot(y, 10, 0)
    elif loss == "sigmoid":
        return y


def load_mnist(loss: str) -> (TensorDataset, TensorDataset):
    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data / 255), flatten(mnist_test.data / 255)
    y_train, y_test = make_labels(mnist_train.targets, loss), \
        make_labels(mnist_test.targets, loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(unflatten(standardized_X_train, (1, 28, 28)).float(), y_train)
    test = TensorDataset(unflatten(standardized_X_test, (1, 28, 28)).float(), y_test)
    return train, test



