# ==================== File: locally_connected.py ====================
"""
Locally connected layer for NOTEARS‑MLP (each output variable has its own MLP).
"""
import torch
import torch.nn as nn
import math

class LocallyConnected(nn.Module):
    def __init__(self, num_vars, in_features, out_features, bias=True):
        super().__init__()
        self.num_vars = num_vars
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(num_vars, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_vars, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [n, d, m1]
        n, d, _ = x.shape
        out = torch.einsum('ndi,doi->ndo', x, self.weight)
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
