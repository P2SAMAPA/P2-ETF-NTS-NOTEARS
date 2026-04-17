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
        self._loss = None
        self._grad = None

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def assign_bounds(self, model):
        """Assign bounds for all trainable parameters of the NTS_NOTEARS model."""
        bounds = []

        def add_unbounded(count):
            for _ in range(count):
                bounds.append((None, None))

        for conv in [model.conv1d_pos, model.conv1d_neg]:
            if hasattr(conv, 'instantaneous_bounds'):
                bounds.extend(conv.instantaneous_bounds)
            if hasattr(conv, 'lag_bounds_lists'):
                for var_bounds in conv.lag_bounds_lists:
                    bounds.extend(var_bounds)
            if conv.bias is not None:
                add_unbounded(conv.bias.numel())

        for fc in model.fc2:
            add_unbounded(fc.weight.numel())
            if fc.bias is not None:
                add_unbounded(fc.bias.numel())

        self.bounds = bounds
        total_params = sum(p.numel() for p in self.params)
        if len(self.bounds) != total_params:
            raise ValueError(f"Bounds length {len(self.bounds)} does not match parameter count {total_params}")

    def step(self, closure):
        """Run one L‑BFGS‑B step. `closure` should compute loss and populate gradients."""
        self.closure = closure

        def scipy_objective(x):
            self._set_params(x)
            self.zero_grad()
            loss = self.closure()
            self._loss = loss.item()
            self._grad = self._get_grads()
            return self._loss

        def scipy_grad(x):
            # Use the pre‑computed gradient from the last objective call
            return self._grad

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
        return torch.cat([g.view(-1) for g in grads]).cpu().numpy().astype(np.float64)
