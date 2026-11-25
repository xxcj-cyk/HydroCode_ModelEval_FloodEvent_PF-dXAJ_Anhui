"""HYMOD conceptual rainfall-runoff model (MARRMoT m_29 formulation)."""

import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


@jit(nopython=True)
def simulate(p_and_e, cmax, bexp, alpha, ks, kq):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    qsim = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    sm = 0.5 * cmax.copy()
    sf1 = np.zeros(n_basin)
    sf2 = np.zeros(n_basin)
    sf3 = np.zeros(n_basin)
    slow = np.zeros(n_basin)

    for t in range(n_time):
        for j in range(n_basin):
            p = p_and_e[t, j, 0]
            ep = p_and_e[t, j, 1]
            if p < 0.0:
                p = 0.0
            if ep < 0.0:
                ep = 0.0

            cm = cmax[j]
            if cm <= 0.0:
                cm = 1.0
            wet = sm[j] / cm
            if wet < 0.0:
                wet = 0.0
            if wet > 1.0:
                wet = 1.0

            ea = wet * ep
            pe = (1.0 - (1.0 - wet) ** bexp[j]) * p

            sm[j] = sm[j] + p - ea - pe
            if sm[j] < 0.0:
                sm[j] = 0.0
            if sm[j] > cm:
                sm[j] = cm
            ets[t, j] = ea

            pf = alpha[j] * pe
            ps = (1.0 - alpha[j]) * pe

            sf1[j] = sf1[j] + pf
            qf1 = kq[j] * sf1[j]
            sf1[j] = sf1[j] - qf1
            sf2[j] = sf2[j] + qf1
            qf2 = kq[j] * sf2[j]
            sf2[j] = sf2[j] - qf2
            sf3[j] = sf3[j] + qf2
            qf3 = kq[j] * sf3[j]
            sf3[j] = sf3[j] - qf3

            qs = ks[j] * slow[j]
            slow[j] = slow[j] + ps - qs
            if slow[j] < 0.0:
                slow[j] = 0.0

            qsim[t, j] = qf3 + qs
            if qsim[t, j] < 0.0:
                qsim[t, j] = 0.0
    return qsim, ets


def hymod(p_and_e, parameters, warmup_length=365):
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
