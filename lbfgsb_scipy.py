# ==================== File: lbfgsb_scipy.py ====================
"""
L‑BFGS‑B optimizer wrapper for PyTorch.
"""
import torch
import scipy.optimize
import numpy as np

class LBFGSBScipy:
    def __init__(self, params):
        self.params = list(params)
        self.bounds = None

    def assign_bounds(self, model):
        """Assign bounds from model's convolutional layers."""
        bounds = []
        # Instantaneous bounds for pos and neg conv layers
        if hasattr(model.conv1d_pos, 'instantaneous_bounds'):
            bounds.extend(model.conv1d_pos.instantaneous_bounds)
        if hasattr(model.conv1d_neg, 'instantaneous_bounds'):
            bounds.extend(model.conv1d_neg.instantaneous_bounds)
        # Lag bounds (flattened)
        for bound_list in [model.conv1d_pos.lag_bounds_lists, model.conv1d_neg.lag_bounds_lists]:
            for var_bounds in bound_list:
                bounds.extend(var_bounds)
        # FC2 layers are unconstrained
        for _ in model.fc2.parameters():
            bounds.append((None, None))
        self.bounds = bounds

    def step(self, closure):
        def scipy_objective(x):
            self._set_params(x)
            loss = closure()
            return loss.item()

        def scipy_grad(x):
            self._set_params(x)
            closure().backward()
            grad = self._get_grads()
            return grad.cpu().numpy().astype(np.float64)

        x0 = self._get_params()
        result = scipy.optimize.minimize(
            scipy_objective, x0, method='L-BFGS-B', jac=scipy_grad,
            bounds=self.bounds, options={'maxiter': 100}
        )
        self._set_params(result.x)

    def _get_params(self):
        return torch.cat([p.data.view(-1) for p in self.params]).cpu().numpy()

    def _set_params(self, x):
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data = torch.tensor(x[offset:offset+numel], dtype=p.dtype).view_as(p)
            offset += numel

    def _get_grads(self):
        grads = []
        for p in self.params:
            if p.grad is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.data.clone())
        return torch.cat([g.view(-1) for g in grads])
