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
        """Assign bounds for all trainable parameters of the NTS_NOTEARS model."""
        bounds = []

        # Helper to add (None, None) for any remaining parameters (e.g., biases)
        def add_unbounded(count):
            for _ in range(count):
                bounds.append((None, None))

        # 1. Convolutional layers: pos and neg
        for conv in [model.conv1d_pos, model.conv1d_neg]:
            # Instantaneous bounds (kernel_size index = n_lags)
            if hasattr(conv, 'instantaneous_bounds'):
                bounds.extend(conv.instantaneous_bounds)
            # Lag bounds (list of lists)
            if hasattr(conv, 'lag_bounds_lists'):
                for var_bounds in conv.lag_bounds_lists:
                    bounds.extend(var_bounds)
            # Bias terms (if any) are unconstrained
            if conv.bias is not None:
                add_unbounded(conv.bias.numel())

        # 2. LocallyConnected layers (fc2)
        for fc in model.fc2:
            # weight
            bounds.extend([(None, None)] * fc.weight.numel())
            # bias (if present)
            if fc.bias is not None:
                bounds.extend([(None, None)] * fc.bias.numel())

        self.bounds = bounds

        # Sanity check
        total_params = sum(p.numel() for p in self.params)
        if len(self.bounds) != total_params:
            raise ValueError(f"Bounds length {len(self.bounds)} does not match parameter count {total_params}")

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
