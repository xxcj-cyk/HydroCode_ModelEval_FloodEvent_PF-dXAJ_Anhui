import numpy as np
from numba import jit

from .XAJ import (
    calc_runoff_series,
    constrain_ki_kg,
    initial_storage_from_forcing,
    route_linear,
)
from hydromodels_pbm.model.mz import (
    convolve_truncated,
    gamma_uh_weights,
    kernel_length,
)
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_mz(runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0):
    n_time, n_basin = ris.shape
    inp = runoff_im + rss
    qs_surface = np.zeros((n_time, n_basin))
    length = kernel_length(n_time)
    for j in range(n_basin):
        w = gamma_uh_weights(a[j], theta[j], length)
        qs_surface[:, j] = convolve_truncated(inp[:, j], w)
    return route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_xaj_mz(p_and_e, parameters, warmup_length=365):
    if p_and_e.shape[2] != 2:
        raise ValueError(
            'XAJ-mz expects drivers [time, basin, 2]: P, PET '
            f'(got feature dim {p_and_e.shape[2]})'
        )
    params = constrain_ki_kg(to_array_2d(parameters))
    k = params[:, 0]
    b = params[:, 1]
    im = params[:, 2]
    um = params[:, 3]
    lm = params[:, 4]
    dm = params[:, 5]
    c = params[:, 6]
    sm = params[:, 7]
    ex = params[:, 8]
    ki = params[:, 9]
    kg = params[:, 10]
    a = params[:, 11]
    theta = params[:, 12]
    ci = params[:, 13]
    cg = params[:, 14]

    wu, wl, wd, s = initial_storage_from_forcing(
        p_and_e, um, lm, dm, sm, warmup_length=warmup_length
    )
    n_basin = p_and_e.shape[1]
    fr = np.full(n_basin, 0.1)
    qi0 = np.full(n_basin, 0.1)
    qg0 = np.full(n_basin, 0.1)

    runoff_im, rss, ris, rgs, ets = calc_runoff_series(
        p_and_e, k, b, im, um, lm, dm, c, sm, ex, ki, kg, wu, wl, wd, s, fr
    )
    qs = route_mz(
        runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0
    )
    return qs, ets


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def xaj_mz(p_and_e, parameters, warmup_length=365):
    qsim, ets = simulate_xaj_mz(
        p_and_e, parameters, warmup_length=warmup_length
    )
    return trim_warmup_output(qsim, ets, warmup_length)
