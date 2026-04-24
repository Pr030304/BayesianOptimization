from __future__ import annotations

import numpy as np


domain_bounds = [[0.0, 1.0]] * 4
maximize = True
init_method = "latin_hc"
init_capital = "default"
metric_warmup_evals = 15
metric_update_subset = "topk"
metric_subset_size = 8
metric_shrinkage = 0.02
MU = np.array([0.68, 0.27, 0.61, 0.79], dtype=float)

A = np.array(
    [
        [1.0, 0.9, 0.3, 0.2],
        [0.2, 1.0, 0.8, 0.1],
        [0.1, 0.4, 1.0, 0.7],
        [0.3, 0.2, 0.5, 1.0],
    ],
    dtype=float,
)
Q, _ = np.linalg.qr(A)
R = Q

true_opt_point = MU.tolist()
true_opt_value = 1.0


def objective(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    z = R.T @ (x - MU)

    # Rotated 2D active subspace:
    # narrow in z[1], broader in z[0], weak dependence on the others.
    q = (
        (z[0] / 0.32) ** 2
        + (z[1] / 0.045) ** 2
        + 0.08 * (z[2] / 0.40) ** 4
        + 0.05 * (z[3] / 0.40) ** 4
    )

    return float(1.0 / (1.0 + q))




# def objective(x: np.ndarray) -> float:
#     x = np.asarray(x, dtype=float)

#     alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
#     A = np.array(
#         [
#             [3.0, 10.0, 30.0],
#             [0.1, 10.0, 35.0],
#             [3.0, 10.0, 30.0],
#             [0.1, 10.0, 35.0],
#         ],
#         dtype=float,
#     )
#     P = 1e-4 * np.array(
#         [
#             [3689, 1170, 2673],
#             [4699, 4387, 7470],
#             [1091, 8732, 5547],
#             [381, 5743, 8828],
#         ],
#         dtype=float,
#     )

#     inner = np.sum(A * (x - P) ** 2, axis=1)
#     return float(np.sum(alpha * np.exp(-inner)))
