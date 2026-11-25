import numpy as np
from numba import jit

from hydromodels_pbm.model.GR4J.GR4J import production_step, unit_hydrograph_weights
from hydromodels_pbm.model.mz import convolve_truncated
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


# ---------------------------------------------------------------------------
# Routing store (GR5J exchange: X2 * R/X3 - X2 * X5)
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_store_step(q9, q1, x2, x3, x5, r):
    if r < 0.0:
        r = 0.0
    if r > x3:
        r = x3
    exchange = x2 * r / x3 - x2 * x5
    r = r + q9 + exchange
    if r < 0.0:
        r = 0.0
    qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
    r = r - qr
    qd = q1 + exchange
    if qd < 0.0:
        qd = 0.0
    return qr + qd, r


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@jit(nopython=True)
def simulate(p_and_e, x1, x2, x3, x4, x5):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    prs = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    # airGR default IniResLevels: Prod=0.3*X1, Rout=0.5*X3 (CreateRunOptions.R)
    s = 0.3 * x1.copy()
    for t in range(n_time):
        for j in range(n_basin):
            prs[t, j], ets[t, j], s[j] = production_step(
                p_and_e[t, j, 0], p_and_e[t, j, 1], x1[j], s[j]
            )

    # GR4J-family routing: 90% PR -> UH1 (base X4), 10% PR -> UH2 (base 2*X4)
    uh1_frac = 0.9
    q9 = np.zeros((n_time, n_basin))
    q1 = np.zeros((n_time, n_basin))
    for j in range(n_basin):
        w_uh1 = unit_hydrograph_weights(x4[j], True)
        w_uh2 = unit_hydrograph_weights(x4[j], False)
        q9[:, j] = uh1_frac * convolve_truncated(prs[:, j], w_uh1)
        q1[:, j] = (1.0 - uh1_frac) * convolve_truncated(prs[:, j], w_uh2)

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


def gr5j(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim, ets = simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
