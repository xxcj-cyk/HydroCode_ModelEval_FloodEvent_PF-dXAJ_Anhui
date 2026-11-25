"""Sugawara-style four-tank rainfall-runoff model."""

import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


@jit(nopython=True)
def simulate(p_and_e, h1, h2, h3, h4, k1, k2, k3, k4, f12, f23):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    qsim = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    s1 = np.zeros(n_basin)
    s2 = np.zeros(n_basin)
    s3 = np.zeros(n_basin)
    s4 = np.zeros(n_basin)

    for t in range(n_time):
        for j in range(n_basin):
            p = p_and_e[t, j, 0]
            e = p_and_e[t, j, 1]
            if p < 0.0:
                p = 0.0
            if e < 0.0:
                e = 0.0

            ea = e
            if ea > s1[j]:
                ea = s1[j]
            s1[j] = s1[j] - ea
            ets[t, j] = ea
            s1[j] = s1[j] + p

            q1 = k1[j] * max(s1[j] - h1[j], 0.0)
            s1[j] = s1[j] - q1
            tr12 = f12[j] * s1[j]
            s1[j] = s1[j] - tr12
            if s1[j] < 0.0:
                s1[j] = 0.0

            s2[j] = s2[j] + tr12
            q2 = k2[j] * max(s2[j] - h2[j], 0.0)
            s2[j] = s2[j] - q2
            tr23 = f23[j] * s2[j]
            s2[j] = s2[j] - tr23
            if s2[j] < 0.0:
                s2[j] = 0.0

            s3[j] = s3[j] + tr23
            q3 = k3[j] * max(s3[j] - h3[j], 0.0)
            s3[j] = s3[j] - q3
            tr34 = s3[j]
            s3[j] = 0.0

            s4[j] = s4[j] + tr34
            q4 = k4[j] * max(s4[j] - h4[j], 0.0)
            s4[j] = s4[j] - q4
            if s4[j] < 0.0:
                s4[j] = 0.0

            qsim[t, j] = q1 + q2 + q3 + q4
            if qsim[t, j] < 0.0:
                qsim[t, j] = 0.0
    return qsim, ets


def tank(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim, ets = simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
        params[:, 6],
        params[:, 7],
        params[:, 8],
        params[:, 9],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
