import numpy as np
from numba import jit

from hydromodels_pbm.model.GR4J.GR4J import production_step
from hydromodels_pbm.model.mz import (
    convolve_truncated,
    gamma_uh_weights,
    kernel_length,
)
from hydromodels_pbm.model.GR5J.GR5J import route_store_step
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@jit(nopython=True)
def simulate_mz(p_and_e, x1, x2, x3, a, theta, x5):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    prs = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    s = 0.3 * x1.copy()
    for t in range(n_time):
        for j in range(n_basin):
            prs[t, j], ets[t, j], s[j] = production_step(
                p_and_e[t, j, 0], p_and_e[t, j, 1], x1[j], s[j]
            )

    uh1_frac = 0.9
    q9 = np.zeros((n_time, n_basin))
    q1 = np.zeros((n_time, n_basin))
    length = kernel_length(n_time)
    for j in range(n_basin):
        w = gamma_uh_weights(a[j], theta[j], length)
        conv = convolve_truncated(prs[:, j], w)
        q9[:, j] = uh1_frac * conv
        q1[:, j] = (1.0 - uh1_frac) * conv

    qsim = np.zeros((n_time, n_basin))
    r = 0.5 * x3.copy()
    for t in range(n_time):
        for j in range(n_basin):
            qsim[t, j], r[j] = route_store_step(
                q9[t, j], q1[t, j], x2[j], x3[j], x5[j], r[j]
            )
    return qsim, ets


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def gr5j_mz(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim, ets = simulate_mz(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
