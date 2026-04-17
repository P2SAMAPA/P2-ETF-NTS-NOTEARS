# ==================== File: trace_expm.py ====================
"""
Trace of matrix exponential for DAG constraint.
"""
import torch
import numpy as np
from scipy.linalg import expm

def trace_expm(A: torch.Tensor) -> float:
    """Compute trace(expm(A)) - d using scipy.linalg.expm."""
    A_np = A.detach().cpu().numpy()
    return np.trace(expm(A_np))
