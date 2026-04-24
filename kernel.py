from __future__ import annotations

import numpy as np

from .utils import dist_squared


class Kernel:
    def __init__(self):
        self.hyperparams = {}

    def is_guaranteed_psd(self):
        raise NotImplementedError

    def __call__(self, X1, X2=None):
        return self.evaluate(X1, X2)

    def evaluate(self, X1, X2=None):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        if len(X1) == 0 or len(X2) == 0:
            return np.zeros((len(X1), len(X2)))
        return self._child_evaluate(X1, X2)

    def add_hyperparams(self, **kwargs):
        for key, value in kwargs.items():
            self.hyperparams[key] = value

    def gradient(self, param, X1, X2=None, *args):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        if len(X1) == 0 or len(X2) == 0:
            return np.zeros((len(X1), len(X2)))
        return self._child_gradient(param, X1, X2, *args)

    def input_gradient(self, X1, X2=None):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        if len(X1) == 0 or len(X2) == 0:
            return np.zeros((len(X1), len(X2), getattr(self, "dim", 0)))
        return self._child_input_gradient(X1, X2)


class SEKernel(Kernel):
    def __init__(self, dim, scale=None, dim_bandwidths=None):
        super().__init__()
        self.dim = dim
        self.set_se_hyperparams(scale, dim_bandwidths)

    def is_guaranteed_psd(self):
        return True

    def set_se_hyperparams(self, scale, dim_bandwidths):
        self.add_hyperparams(scale=scale)
        dim_bandwidths = np.asarray(dim_bandwidths, dtype=float).reshape(1, -1)
        self.add_hyperparams(dim_bandwidths=dim_bandwidths)

    def get_scaled_repr(self, X):
        return X / self.hyperparams["dim_bandwidths"]

    def _child_evaluate(self, X1, X2):
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        return self.hyperparams["scale"] * np.exp(-dist_squared(scaled_X1, scaled_X2) / 2.0)

    def _child_gradient(self, param, X1, X2, param_num=None):
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        dist_sq = dist_squared(scaled_X1, scaled_X2)
        base = self.hyperparams["scale"] * np.exp(-dist_sq / 2.0)
        if param == "scale":
            return base
        if param == "same_dim_bandwidths":
            dist_sq_dv = dist_sq / self.hyperparams["dim_bandwidths"][0, 0]
            return base * dist_sq_dv
        dim_X1 = np.expand_dims(scaled_X1[:, param_num], axis=1)
        dim_X2 = np.expand_dims(scaled_X2[:, param_num], axis=1)
        dim_sq = dist_squared(dim_X1, dim_X2)
        dim_sq = dim_sq / self.hyperparams["dim_bandwidths"][0, param_num]
        return base * dim_sq

    def _child_input_gradient(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        base = self._child_evaluate(X1, X2)
        bandwidth_sq = (self.hyperparams["dim_bandwidths"] ** 2).reshape(1, 1, -1)
        return -base[:, :, None] * diff / bandwidth_sq


class MatrixSEKernel(Kernel):
    def __init__(self, dim, scale=None, lengthscale=None, metric_matrix=None):
        super().__init__()
        self.dim = dim
        self.set_matrix_se_hyperparams(scale, lengthscale, metric_matrix)

    def is_guaranteed_psd(self):
        return True

    def set_matrix_se_hyperparams(self, scale, lengthscale, metric_matrix):
        metric_matrix = np.asarray(metric_matrix, dtype=float)
        if metric_matrix.shape != (self.dim, self.dim):
            raise ValueError(
                f"metric_matrix must have shape {(self.dim, self.dim)}, got {metric_matrix.shape}"
            )
        metric_matrix = 0.5 * (metric_matrix + metric_matrix.T)
        self.add_hyperparams(
            scale=float(scale),
            lengthscale=float(lengthscale),
            metric_matrix=metric_matrix,
        )

    def _child_evaluate(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        metric_matrix = self.hyperparams["metric_matrix"]
        quad_form = np.einsum("...i,ij,...j->...", diff, metric_matrix, diff)
        exponent = -quad_form / (2.0 * (self.hyperparams["lengthscale"] ** 2))
        return self.hyperparams["scale"] * np.exp(exponent)

    def _child_gradient(self, param, X1, X2, param_num=None):
        base = self._child_evaluate(X1, X2)
        if param == "scale":
            return base / self.hyperparams["scale"]
        if param == "lengthscale":
            diff = X1[:, None, :] - X2[None, :, :]
            metric_matrix = self.hyperparams["metric_matrix"]
            quad_form = np.einsum("...i,ij,...j->...", diff, metric_matrix, diff)
            return base * quad_form / (self.hyperparams["lengthscale"] ** 3)
        raise ValueError(f"Unsupported parameter for MatrixSEKernel gradient: {param}")

    def _child_input_gradient(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        base = self._child_evaluate(X1, X2)
        metric_diff = np.einsum("ij,...j->...i", self.hyperparams["metric_matrix"], diff)
        lengthscale_sq = self.hyperparams["lengthscale"] ** 2
        return -base[:, :, None] * metric_diff / lengthscale_sq
