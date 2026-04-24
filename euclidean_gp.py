from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp_core
from .kernel import MatrixSEKernel, SEKernel


class EuclideanGP(gp_core.GP):
    def __init__(self, X, Y, kernel, mean_func, noise_var, build_posterior=True):
        super().__init__(X, Y, kernel, mean_func, noise_var, build_posterior)


@dataclass
class EuclideanGPFitterConfig:
    kernel_type: str = "se"
    mean_func_type: str = "tune"
    mean_func_const: float = 0.0
    noise_var_type: str = "tune"
    noise_var_label: float = 0.05
    noise_var_value: float = 0.1
    hp_tune_max_evals: int = -1
    ml_hp_tune_opt: str = "rand"
    use_same_bandwidth: bool = False
    metric_matrix: np.ndarray | None = None
    rng: np.random.Generator | None = None


def get_se_kernel(dim, gp_cts_hps, use_same_bandwidth):
    scale = np.exp(gp_cts_hps[0])
    gp_cts_hps = gp_cts_hps[1:]
    if use_same_bandwidth:
        dim_bandwidths = [np.exp(gp_cts_hps[0])] * dim
        gp_cts_hps = gp_cts_hps[1:]
    else:
        dim_bandwidths = np.exp(gp_cts_hps[:dim])
        gp_cts_hps = gp_cts_hps[dim:]
    kernel = SEKernel(dim=dim, scale=scale, dim_bandwidths=dim_bandwidths)
    return kernel, gp_cts_hps


def get_matrix_se_kernel(dim, gp_cts_hps, metric_matrix):
    scale = np.exp(gp_cts_hps[0])
    lengthscale = np.exp(gp_cts_hps[1])
    gp_cts_hps = gp_cts_hps[2:]
    kernel = MatrixSEKernel(
        dim=dim,
        scale=scale,
        lengthscale=lengthscale,
        metric_matrix=metric_matrix,
    )
    return kernel, gp_cts_hps


class EuclideanGPFitter:
    def __init__(self, X, Y, options=None):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.dim = self.X.shape[1]
        self.options = options or EuclideanGPFitterConfig()
        if self.options.kernel_type not in ["se", "matrix_se"]:
            raise ValueError(f"Unsupported kernel_type: {self.options.kernel_type}")
        if self.options.noise_var_type not in ["tune", "label", "value"]:
            raise ValueError(f"Unsupported noise_var_type: {self.options.noise_var_type}")
        if self.options.mean_func_type not in ["mean", "median", "const", "zero", "tune"]:
            raise ValueError(f"Unsupported mean_func_type: {self.options.mean_func_type}")
        self.Y_var = np.array(self.Y).std() ** 2 + 1e-4
        self.cts_hp_bounds = []
        self.rng = self.options.rng if self.options.rng is not None else np.random.default_rng()
        self._set_up()

    def _set_up(self):
        self._set_up_mean_and_noise_variance_bounds()
        self._set_up_kernel_bounds()
        self.cts_hp_bounds = np.asarray(self.cts_hp_bounds, dtype=float)
        self.hp_tune_max_evals = self._resolve_hp_tune_max_evals()

    def _set_up_mean_and_noise_variance_bounds(self):
        if self.options.mean_func_type == "tune":
            Y_std = np.sqrt(self.Y_var)
            Y_median = np.median(self.Y) if len(self.Y) > 0 else 0.0
            Y_half_range = 0.5 * (max(self.Y) - min(self.Y)) if len(self.Y) > 0 else 1.0
            Y_width = 0.5 * (Y_half_range + Y_std)
            self.cts_hp_bounds.append([Y_median - 3 * Y_width, Y_median + 3 * Y_width])
        if self.options.noise_var_type == "tune":
            self.cts_hp_bounds.append([np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)])

    def _set_up_kernel_bounds(self):
        self.cts_hp_bounds.append([np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)])
        X_std_norm = np.linalg.norm(self.X, "fro") + 1e-4
        bw_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
        if self.options.kernel_type == "matrix_se":
            self.cts_hp_bounds.append(bw_bounds)
        else:
            if self.options.use_same_bandwidth:
                self.cts_hp_bounds.append(bw_bounds)
            else:
                for _ in range(self.dim):
                    self.cts_hp_bounds.append(bw_bounds)

    def _resolve_hp_tune_max_evals(self):
        if self.options.hp_tune_max_evals is not None and self.options.hp_tune_max_evals > 0:
            return int(self.options.hp_tune_max_evals)
        num_hps = len(self.cts_hp_bounds)
        if self.options.ml_hp_tune_opt == "rand":
            return int(min(1e4, max(500, num_hps * 200)))
        return int(min(1e4, max(500, num_hps * 50)))

    def _build_mean_func_and_noise_var(self, gp_cts_hps):
        idx = 0
        if self.options.mean_func_type == "mean":
            mean_value = float(np.mean(self.Y))
        elif self.options.mean_func_type == "median":
            mean_value = float(np.median(self.Y))
        elif self.options.mean_func_type == "const":
            mean_value = float(self.options.mean_func_const)
        elif self.options.mean_func_type == "zero":
            mean_value = 0.0
        elif self.options.mean_func_type == "tune":
            mean_value = gp_cts_hps[idx].item()
            idx += 1
        else:
            raise ValueError(f"Unsupported mean_func_type: {self.options.mean_func_type}")
        mean_func = lambda x, mean_value=mean_value: np.asarray([mean_value] * len(x))
        if self.options.noise_var_type == "tune":
            noise_var = np.exp(gp_cts_hps[idx])
            idx += 1
        elif self.options.noise_var_type == "label":
            noise_var = float(self.options.noise_var_label) * (self.Y.std() ** 2)
        elif self.options.noise_var_type == "value":
            noise_var = float(self.options.noise_var_value)
        else:
            raise ValueError(f"Unsupported noise_var_type: {self.options.noise_var_type}")
        return mean_func, noise_var, gp_cts_hps[idx:]

    def build_gp(self, gp_cts_hps):
        mean_func, noise_var, kernel_cts_hps = self._build_mean_func_and_noise_var(np.asarray(gp_cts_hps))
        if self.options.kernel_type == "matrix_se":
            metric_matrix = self.options.metric_matrix
            if metric_matrix is None:
                metric_matrix = np.eye(self.dim)
            kernel, _ = get_matrix_se_kernel(self.dim, kernel_cts_hps, metric_matrix)
        else:
            kernel, _ = get_se_kernel(self.dim, kernel_cts_hps, self.options.use_same_bandwidth)
        return EuclideanGP(self.X, self.Y, kernel, mean_func, noise_var)

    def _tuning_objective(self, gp_cts_hps):
        built_gp = self.build_gp(gp_cts_hps)
        return built_gp.compute_log_marginal_likelihood()

    def _sample_cts_hps(self, n_samples):
        lows = self.cts_hp_bounds[:, 0]
        highs = self.cts_hp_bounds[:, 1]
        return self.rng.uniform(lows, highs, size=(n_samples, len(self.cts_hp_bounds)))

    def fit_gp(self):
        candidate_cts_hps = self._sample_cts_hps(self.hp_tune_max_evals)
        best_val = -np.inf
        best_gp = None
        best_hps = None
        for cts_hps in candidate_cts_hps:
            val = self._tuning_objective(cts_hps)
            if val > best_val:
                best_val = val
                best_gp = self.build_gp(cts_hps)
                best_hps = cts_hps
        return "fitted_gp", best_gp, best_hps
