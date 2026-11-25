"""HBV-light style lumped conceptual rainfall-runoff model (no snow routine)."""

import math

import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


@jit(nopython=True)
def triangle_weights(mb):
    mb_int = int(math.ceil(mb))
    if mb_int < 1:
        mb_int = 1
    w = np.zeros(mb_int)
    half = mb_int // 2
    total = 0.0
    for i in range(mb_int):
        t = i + 1
        if t <= half:
            w[i] = float(t)
        else:
            w[i] = float(mb_int - t + 1)
        total += w[i]
    for i in range(mb_int):
        w[i] /= total
    return w


@jit(nopython=True)
def route_triangle(q, maxbas):
    n_time = q.shape[0]
    n_basin = q.shape[1]
    out = np.zeros((n_time, n_basin))
    for j in range(n_basin):
        mb = int(math.ceil(maxbas[j]))
        if mb < 1:
            mb = 1
        w = triangle_weights(float(mb))
        m = w.shape[0]
        for t in range(n_time):
            acc = 0.0
            for k in range(m):
                idx = t - k
                if idx >= 0:
                    acc += q[idx, j] * w[k]
            out[t, j] = acc
    return out


@jit(nopython=True)
def simulate(p_and_e, fc, beta, lp, perc, uzl, k0, k1, k2, maxbas):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    qsim = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    sm = 0.5 * fc.copy()
    uz = np.zeros(n_basin)
    lz = np.zeros(n_basin)

    for t in range(n_time):
        for j in range(n_basin):
            p = p_and_e[t, j, 0]
            pet = p_and_e[t, j, 1]
            if p < 0.0:
                p = 0.0
            if pet < 0.0:
                pet = 0.0

            denom = lp[j] * fc[j]
            if denom <= 0.0:
                ea = 0.0
            else:
                ea = pet * sm[j] / denom
                if ea > sm[j]:
                    ea = sm[j]
                if ea > pet:
                    ea = pet
            sm[j] = sm[j] + p - ea
            ets[t, j] = ea
            if sm[j] < 0.0:
                sm[j] = 0.0

            if sm[j] > fc[j]:
                recharge = sm[j] - fc[j]
                sm[j] = fc[j]
            else:
                wet = sm[j] / fc[j] if fc[j] > 0.0 else 0.0
                if wet < 0.0:
                    wet = 0.0
                if wet > 1.0:
                    wet = 1.0
                recharge = p * (wet ** beta[j])
                sm[j] = sm[j] - recharge
                if sm[j] < 0.0:
                    sm[j] = 0.0

            uz[j] = uz[j] + recharge
            q0 = 0.0
            if uz[j] > uzl[j]:
                q0 = k0[j] * (uz[j] - uzl[j])
            q1 = k1[j] * uz[j]
            percol = perc[j]
            if percol > uz[j]:
                percol = uz[j]
            uz[j] = uz[j] - q0 - q1 - percol
            if uz[j] < 0.0:
                uz[j] = 0.0

            lz[j] = lz[j] + percol
            q2 = k2[j] * lz[j]
            lz[j] = lz[j] - q2
            if lz[j] < 0.0:
                lz[j] = 0.0

            qsim[t, j] = q0 + q1 + q2
            if qsim[t, j] < 0.0:
                qsim[t, j] = 0.0

    qsim = route_triangle(qsim, maxbas)
    return qsim, ets


def hbv(p_and_e, parameters, warmup_length=365):
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
    )
    return trim_warmup_output(qsim, ets, warmup_length)
