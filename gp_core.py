from __future__ import annotations

import numpy as np

from .utils import (
    draw_gaussian_samples,
    project_symmetric_to_psd_cone,
    solve_lower_triangular,
    solve_upper_triangular,
    stable_cholesky,
)


def _check_feature_label_lengths_and_format(X, Y):
    if len(X) != len(Y):
        raise ValueError(f"Length mismatch: len(X)={len(X)} len(Y)={len(Y)}")


def _get_cholesky_decomp(K_trtr_wo_noise, noise_var, handle_non_psd_kernels):
    if handle_non_psd_kernels == "try_before_project":
        K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
        try:
            return stable_cholesky(K_trtr_w_noise, add_to_diag_till_psd=False)
        except np.linalg.LinAlgError:
            return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, "project_first")
    if handle_non_psd_kernels == "project_first":
        K_trtr_wo_noise = project_symmetric_to_psd_cone(K_trtr_wo_noise)
        return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, "guaranteed_psd")
    if handle_non_psd_kernels == "guaranteed_psd":
        K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
        return stable_cholesky(K_trtr_w_noise)
    raise ValueError(f"Unknown option for handle_non_psd_kernels: {handle_non_psd_kernels}")


def get_post_covar_from_raw_covar(raw_post_covar, noise_var, is_guaranteed_psd):
    if is_guaranteed_psd:
        return raw_post_covar
    epsilon = 0.05 * noise_var
    return project_symmetric_to_psd_cone(raw_post_covar, epsilon=epsilon)


class GP:
    def __init__(
        self,
        X,
        Y,
        kernel,
        mean_func,
        noise_var,
        build_posterior=True,
        handle_non_psd_kernels="guaranteed_psd",
    ):
        _check_feature_label_lengths_and_format(X, Y)
        self.set_data(X, Y, build_posterior=False)
        self.kernel = kernel
        self.mean_func = mean_func
        self.noise_var = noise_var
        self.handle_non_psd_kernels = handle_non_psd_kernels
        self.L = None
        self.alpha = None
        self.K_trtr_wo_noise = None
        self._set_up()
        if build_posterior:
            self.build_posterior()

    def _set_up(self):
        if not self.kernel.is_guaranteed_psd():
            assert self.handle_non_psd_kernels in ["project_first", "try_before_project"]

    def set_data(self, X, Y, build_posterior=True):
        self.X = list(X)
        self.Y = np.asarray(Y, dtype=float)
        self.num_tr_data = len(self.Y)
        if build_posterior:
            self.build_posterior()

    def add_data_multiple(self, X_new, Y_new, build_posterior=True):
        _check_feature_label_lengths_and_format(X_new, Y_new)
        self.X.extend(X_new)
        self.Y = np.concatenate([self.Y, np.asarray(Y_new, dtype=float)])
        self.num_tr_data = len(self.Y)
        if build_posterior:
            self.build_posterior()

    def _get_training_kernel_matrix(self):
        return self.kernel(self.X, self.X)

    def build_posterior(self):
        self.K_trtr_wo_noise = self._get_training_kernel_matrix()
        self.L = _get_cholesky_decomp(
            self.K_trtr_wo_noise, self.noise_var, self.handle_non_psd_kernels
        )
        Y_centred = self.Y - self.mean_func(self.X)
        self.alpha = solve_upper_triangular(
            self.L.T, solve_lower_triangular(self.L, Y_centred)
        )

    def eval(self, X_test, uncert_form="none"):
        X_test = np.asarray(X_test, dtype=float)
        test_mean = self.mean_func(X_test)
        K_tetr = self.kernel(X_test, self.X)
        pred_mean = test_mean + K_tetr.dot(self.alpha)
        if uncert_form == "none":
            return pred_mean, None
        K_tete = self.kernel(X_test, X_test)
        V = solve_lower_triangular(self.L, K_tetr.T)
        post_covar = K_tete - V.T.dot(V)
        post_covar = get_post_covar_from_raw_covar(
            post_covar, self.noise_var, self.kernel.is_guaranteed_psd()
        )
        if uncert_form == "covar":
            return pred_mean, post_covar
        if uncert_form == "std":
            return pred_mean, np.sqrt(np.diag(post_covar))
        raise ValueError("uncert_form should be none, covar or std.")

    def compute_log_marginal_likelihood(self):
        Y_centred = self.Y - self.mean_func(self.X)
        return (
            -0.5 * Y_centred.T.dot(self.alpha)
            - np.log(np.diag(self.L)).sum()
            - 0.5 * self.num_tr_data * np.log(2 * np.pi)
        )

    def eval_mean_gradient(self, x):
        x = np.asarray(x, dtype=float)
        x_is_vector = x.ndim == 1
        x_eval = x.reshape(1, -1) if x_is_vector else x
        try:
            kernel_grad = self.kernel.input_gradient(x_eval, self.X)
        except NotImplementedError:
            return self.eval_mean_gradient_numeric(x)
        pred_grad = np.tensordot(kernel_grad, self.alpha, axes=([1], [0]))
        return pred_grad[0] if x_is_vector else pred_grad

    def eval_mean_gradient_numeric(self, x, step_size=1e-5):
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        for idx in range(len(x)):
            step = step_size * max(1.0, abs(x[idx]))
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += step
            x_minus[idx] -= step
            mu_plus, _ = self.eval(x_plus.reshape(1, -1), uncert_form="none")
            mu_minus, _ = self.eval(x_minus.reshape(1, -1), uncert_form="none")
            grad[idx] = (mu_plus[0] - mu_minus[0]) / (2.0 * step)
        return grad

    def draw_samples(self, num_samples, X_test=None, mean_vals=None, covar=None):
        if X_test is not None:
            mean_vals, covar = self.eval(X_test, "covar")
        return draw_gaussian_samples(num_samples, mean_vals, covar)
