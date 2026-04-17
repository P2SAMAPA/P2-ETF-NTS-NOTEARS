# ==================== File: utils.py ====================
"""
Utility functions.
"""
import torch
import numpy as np

def squared_loss(output, target):
    n = target.shape[0]
    return 0.5 / n * torch.sum((output - target) ** 2)

def reshape_for_model_forward(X, model):
    """X: [n, d] -> [1, d, n]"""
    return X.T.reshape(1, model.dims[0], -1)

def convert_provided_bound(bound_provided_by_user, number_of_hidden_neurons_connected):
    import math
    return math.sqrt(bound_provided_by_user ** 2 / number_of_hidden_neurons_connected)
