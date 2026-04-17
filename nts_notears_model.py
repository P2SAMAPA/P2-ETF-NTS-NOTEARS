# ==================== File: nts_notears_model.py ====================
"""
NTS‑NOTEARS model: CNN + locally connected MLP.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from trace_expm import trace_expm
from utils import convert_provided_bound

class NTS_NOTEARS(nn.Module):
    def __init__(self, dims, n_lags, prior_knowledge=None, variable_names_no_time=None):
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims = dims
        self.n_lags = n_lags
        self.kernel_size = n_lags + 1  # lags + instantaneous
        self.simultaneous_idx = n_lags
        self.prior_knowledge = prior_knowledge
        self.variable_names_no_time = variable_names_no_time
        d = dims[0]

        # 1D convolutions for positive and negative weights
        self.conv1d_pos = nn.Conv1d(d, d * dims[1], bias=True,
                                    kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv1d_neg = nn.Conv1d(d, d * dims[1], bias=True,
                                    kernel_size=self.kernel_size, stride=1, padding=0)

        # Bounds for L‑BFGS‑B (enforce prior knowledge)
        self.conv1d_pos.instantaneous_bounds = self._instantaneous_bounds_pos()
        self.conv1d_neg.instantaneous_bounds = self._instantaneous_bounds_neg()
        self.conv1d_pos.lag_bounds_lists = self._lag_bounds_pos()
        self.conv1d_neg.lag_bounds_lists = self._lag_bounds_neg()

        # Locally connected MLP layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=True))
        self.fc2 = nn.ModuleList(layers)

    def _instantaneous_bounds_pos(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bounds.append((0, 0))
                    else:
                        bounds.append((0, None))
        return bounds

    def _instantaneous_bounds_neg(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    bounds.append((0, 0))
        return bounds

    def _lag_bounds_pos(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    var_bounds = []
                    for lag in range(self.n_lags, 0, -1):
                        var_bounds.append((0, None))
                    bounds.append(var_bounds)
        return bounds

    def _lag_bounds_neg(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    var_bounds = []
                    for lag in range(self.n_lags, 0, -1):
                        var_bounds.append((0, 0))
                    bounds.append(var_bounds)
        return bounds

    def forward(self, x_series):
        # x_series: [1, d, n]
        x = self.conv1d_pos(x_series) - self.conv1d_neg(x_series)  # [1, d*m1, n]
        x = x.T.squeeze(dim=2)                                    # [n, d*m1]
        x = x.view(-1, self.dims[0], self.dims[1])                # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)                                              # [n, d, m2]
        return x.squeeze(dim=2)                                   # [n, d]

    def h_func(self):
        """DAG constraint: trace(expm(A)) - d."""
        d = self.dims[0]
        fc_sim = self.conv1d_pos.weight[:, :, self.simultaneous_idx] - \
                 self.conv1d_neg.weight[:, :, self.simultaneous_idx]
        fc_sim = fc_sim.view(d, -1, d)          # [j, m1, i]
        A = torch.sum(fc_sim * fc_sim, dim=1).t()  # [i, j]
        h = trace_expm(A) - d
        return h

    def l2_reg(self):
        reg = 0.
        reg += torch.sum((self.conv1d_pos.weight - self.conv1d_neg.weight) ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self, kernel_index=None):
        if kernel_index is None:
            return torch.sum(self.conv1d_pos.weight + self.conv1d_neg.weight)
        return torch.sum(self.conv1d_pos.weight[:, :, kernel_index] +
                        self.conv1d_neg.weight[:, :, kernel_index])

    @torch.no_grad()
    def fc1_to_adj(self):
        d = self.dims[0]
        # Instantaneous adjacency
        fc_sim = self.conv1d_pos.weight[:, :, self.simultaneous_idx] - \
                 self.conv1d_neg.weight[:, :, self.simultaneous_idx]
        fc_sim = fc_sim.view(d, -1, d)
        A_sim = torch.sum(fc_sim * fc_sim, dim=1).t()
        W_sim = torch.sqrt(A_sim).cpu().numpy()

        # Lagged adjacency (stacked)
        W_lag = np.empty((0, d))
        for lag_idx in range(self.n_lags):
            fc_lag = self.conv1d_pos.weight[:, :, lag_idx] - \
                     self.conv1d_neg.weight[:, :, lag_idx]
            fc_lag = fc_lag.view(d, -1, d)
            A_lag = torch.sum(fc_lag * fc_lag, dim=1).t()
            W_current = torch.sqrt(A_lag).cpu().numpy()
            W_lag = np.vstack((W_lag, W_current))
        return W_sim, W_lag


# For convenience, re‑export LocallyConnected here or import from separate file
from locally_connected import LocallyConnected
