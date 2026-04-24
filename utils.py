from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def map_to_bounds(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return points * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def get_sublist_from_indices(values, indices):
    return [values[idx] for idx in indices]


def dist_squared(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    n1, dim1 = X1.shape
    n2, dim2 = X2.shape
    if dim1 != dim2:
        raise ValueError("Second dimension of X1 and X2 should be equal.")
    dist_sq = (
        np.outer(np.ones(n1), (X2 ** 2).sum(axis=1))
        + np.outer((X1 ** 2).sum(axis=1), np.ones(n2))
        - 2 * X1.dot(X2.T)
    )
    return np.clip(dist_sq, 0.0, np.inf)


def project_symmetric_to_psd_cone(M: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)
    clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
    return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def stable_cholesky(M: np.ndarray, add_to_diag_till_psd: bool = True) -> np.ndarray:
    if M.size == 0:
        return M
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        if not add_to_diag_till_psd:
            raise exc
        diag_noise_power = -11
        max_M = np.diag(M).max()
        while diag_noise_power < 5:
            diag_noise = (10 ** diag_noise_power) * max_M
            try:
                return np.linalg.cholesky(M + diag_noise * np.eye(M.shape[0]))
            except np.linalg.LinAlgError:
                diag_noise_power += 1
        raise ValueError("Could not compute a stable Cholesky decomposition.")


def solve_lower_triangular(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0 and b.shape[0] == 0:
        return np.zeros(b.shape)
    return solve_triangular(A, b, lower=True)


def solve_upper_triangular(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0 and b.shape[0] == 0:
        return np.zeros(b.shape)
    return solve_triangular(A, b, lower=False)


def draw_gaussian_samples(num_samples: int, mu: np.ndarray, covar: np.ndarray) -> np.ndarray:
    L = stable_cholesky(covar)
    noise = np.random.normal(size=(len(mu), num_samples))
    return (L.dot(noise)).T + mu
