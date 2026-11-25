import math

import numpy as np
from numba import jit

from hydromodels_pbm.model.GR4J.GR4J import production_step
from hydromodels_pbm.model.mz import convolve_truncated
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


# ---------------------------------------------------------------------------
# Unit hydrograph
# ---------------------------------------------------------------------------


@jit(nopython=True)
def s_curve_90(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


@jit(nopython=True)
def s_curve_10(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


@jit(nopython=True)
def unit_hydrograph_weights(x4, quickflow_90):
    if quickflow_90:
        n = int(math.ceil(x4))
    else:
        n = int(math.ceil(2.0 * x4))
    w = np.zeros(n)
    for t in range(1, n + 1):
        if quickflow_90:
            w[t - 1] = s_curve_90(t, x4) - s_curve_90(t - 1, x4)
        else:
            w[t - 1] = s_curve_10(t, x4) - s_curve_10(t - 1, x4)
    return w


# ---------------------------------------------------------------------------
# Exponential store
# ---------------------------------------------------------------------------


@jit(nopython=True)
def exponential_outflow(r2, x6):
    ar = r2 / x6
    if ar < -33.0:
        ar = -33.0
    elif ar > 33.0:
        ar = 33.0
    if ar < -7.0:
        qr2 = x6 * np.exp(ar)
    elif ar > 7.0:
        qr2 = r2 + x6 / np.exp(ar)
    else:
        qr2 = x6 * np.logaddexp(0.0, ar)
    return qr2, r2 - qr2


# ---------------------------------------------------------------------------
# Routing store
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_store_step_gr6(q9, q1, x2, x3, x5, x6, r, r_exp, split_coef):
    if r < 0.0:
        r = 0.0
    if r > x3:
        r = x3
    exchange = x2 * r / x3 - x2 * x5
    r = r + q9 * (1.0 - split_coef) + exchange
    if r < 0.0:
        r = 0.0
    qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
    r = r - qr
    qd = q1 + exchange
    if qd < 0.0:
        qd = 0.0
    q_nl = qr + qd

    # airGR MOD_GR6J: EXCH enters both routing and exponential stores
    r_exp = r_exp + q9 * split_coef + exchange
    qr_exp, r_exp = exponential_outflow(r_exp, x6)
    return q_nl + qr_exp, r, r_exp


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@jit(nopython=True)
def simulate(p_and_e, x1, x2, x3, x4, x5, x6):
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
    for j in range(n_basin):
        w90 = unit_hydrograph_weights(x4[j], True)
        w10 = unit_hydrograph_weights(x4[j], False)
        q9[:, j] = uh1_frac * convolve_truncated(prs[:, j], w90)
        q1[:, j] = (1.0 - uh1_frac) * convolve_truncated(prs[:, j], w10)

    qsim = np.zeros((n_time, n_basin))
    r = 0.5 * x3.copy()
    r_exp = np.zeros(n_basin)
    split_coef = 0.4
    for t in range(n_time):
        for j in range(n_basin):
            qsim[t, j], r[j], r_exp[j] = route_store_step_gr6(
                q9[t, j],
                q1[t, j],
                x2[j],
                x3[j],
                x5[j],
                x6[j],
                r[j],
                r_exp[j],
                split_coef,
            )
    return qsim, ets


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def gr6j(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim, ets = simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
