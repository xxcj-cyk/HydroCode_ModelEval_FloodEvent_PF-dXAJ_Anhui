"""Sherman-type antecedent precipitation index (API) rainfall-runoff model."""

import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


@jit(nopython=True)
def simulate(p_and_e, k, cp, cw):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    qsim = np.zeros((n_time, n_basin))
    w = np.zeros(n_basin)
    for t in range(n_time):
        for j in range(n_basin):
            pn = p_and_e[t, j, 0] - p_and_e[t, j, 1]
            if pn < 0.0:
                pn = 0.0
            w[j] = k[j] * w[j] + pn
            q = cp[j] * pn + cw[j] * w[j]
            if q < 0.0:
                q = 0.0
            qsim[t, j] = q
    return qsim


def api(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim = simulate(p_and_e, params[:, 0], params[:, 1], params[:, 2])
    ets = np.zeros((p_and_e.shape[0], p_and_e.shape[1]), dtype=np.float64)
    return trim_warmup_output(qsim, ets, warmup_length)
