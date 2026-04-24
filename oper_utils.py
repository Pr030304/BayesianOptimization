from __future__ import annotations

from argparse import Namespace

import numpy as np

from .utils import map_to_bounds


def latin_hc_sampling(dim, num_samples, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    num_samples = int(num_samples)
    cut = np.linspace(0.0, 1.0, num_samples + 1)
    u = rng.random((num_samples, dim))
    points = cut[:-1, None] + u * (cut[1:] - cut[:-1])[:, None]
    for dim_idx in range(dim):
        rng.shuffle(points[:, dim_idx])
    return points


def random_sample(obj, bounds, max_evals, vectorised=True, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dim = len(bounds)
    rand_pts = map_to_bounds(rng.random((int(max_evals), dim)), np.asarray(bounds, dtype=float))
    if vectorised:
        obj_vals = obj(rand_pts)
    else:
        obj_vals = np.asarray([obj(x) for x in rand_pts])
    return rand_pts, obj_vals


def random_maximise(obj, bounds, max_evals, return_history=False, vectorised=True, rng=None):
    rand_pts, obj_vals = random_sample(
        obj,
        bounds,
        max_evals,
        vectorised=vectorised,
        rng=rng,
    )
    max_idx = obj_vals.argmax()
    max_val = obj_vals[max_idx]
    max_pt = rand_pts[max_idx]
    history = Namespace(query_vals=obj_vals, query_points=rand_pts) if return_history else None
    return max_val, max_pt, history
