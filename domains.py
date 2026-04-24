from __future__ import annotations

import numpy as np


class EuclideanDomain:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, dtype=float)
        self.diameter = np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0])
        self.dim = len(self.bounds)

    def get_type(self) -> str:
        return "euclidean"

    def get_dim(self) -> int:
        return self.dim

    def is_a_member(self, point) -> bool:
        point = np.asarray(point, dtype=float)
        return np.all(point >= self.bounds[:, 0]) and np.all(point <= self.bounds[:, 1])
