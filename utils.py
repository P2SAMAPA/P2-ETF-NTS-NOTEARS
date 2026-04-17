"""
Utility functions.
"""
import torch
import numpy as np

def squared_loss(output, target):
    n = target.shape[0]
    return 0.5 / n * torch.sum((output - target) ** 2)

def reshape_for_model_forward(X, model, device='cpu'):
    """
    X: numpy array of shape [n, d]
    Returns: torch.Tensor of shape [1, d, n] on the specified device.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    # Transpose to [d, n] then add batch dimension -> [1, d, n]
    return X_tensor.T.unsqueeze(0)

def convert_provided_bound(bound_provided_by_user, number_of_hidden_neurons_connected):
    import math
    return math.sqrt(bound_provided_by_user ** 2 / number_of_hidden_neurons_connected)
