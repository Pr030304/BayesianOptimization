from __future__ import annotations

import numpy as np

from .oper_utils import random_maximise


def _get_ucb_beta_th(dim, time_step):
    return np.sqrt(0.5 * dim * np.log(2 * dim * time_step + 1))


def maximise_acquisition(acq_fn, anc_data):
    if anc_data.domain.get_type() != "euclidean":
        raise ValueError("Only euclidean domains are supported.")
    _, opt_pt, _ = random_maximise(
        acq_fn,
        anc_data.domain.bounds,
        anc_data.max_evals,
        vectorised=True,
        rng=getattr(anc_data, "rng", None),
    )
    return opt_pt


def asy_ucb(gp, anc_data):
    beta_th = _get_ucb_beta_th(gp.kernel.dim, anc_data.t)

    def _ucb_acq(x):
        mu, sigma = gp.eval(x, "std")
        return mu + beta_th * sigma

    return maximise_acquisition(_ucb_acq, anc_data)
