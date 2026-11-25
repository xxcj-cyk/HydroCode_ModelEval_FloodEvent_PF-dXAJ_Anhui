"""Calibration objectives for SCE-UA (per flood event; minimized).

Baselines: ``RMSE``, ``NSE``.

PFLoss (``PF_PEAK80``; knobs via script ``loss_params``)::

    L = overall_w · RMSE + win_w · High
    High = HighUnder if peak_underonly else HighRMSE   (Qobs >= peak_ratio·Qp)
    if peak_excess: High = relu(High − RMSE)

Always uses raw discharge units (no peak-normalization of loss terms).
"""

from __future__ import annotations

import numpy as np

from hydromodels_pbm.workflow.metrics import METRIC_REGISTRY, valid_obs_sim


def rmse_loss(observed, simulated):
    value = METRIC_REGISTRY['RMSE'](simulated, observed)
    if value is None or not np.isfinite(value):
        raise ValueError('RMSE is nan/inf; check input or simulation.')
    return float(value)


def nse_loss(observed, simulated):
    value = METRIC_REGISTRY['NSE'](simulated, observed)
    if value is None or not np.isfinite(value):
        raise ValueError('NSE is nan/inf; check input or simulation.')
    return 1.0 - float(value)


def pf_peak80_loss(
    observed,
    simulated,
    overall_w=1.0,
    win_w=0.25,
    peak_ratio=0.8,
    peak_excess=False,
    peak_underonly=False,
):
    obs, sim = valid_obs_sim(observed, simulated, min_samples=1)
    if obs is None:
        raise ValueError('no valid samples for loss')

    overall = float(np.sqrt(np.mean((sim - obs) ** 2)))
    peak = float(np.max(obs))
    mask = obs >= peak * float(peak_ratio)
    if peak_underonly:
        under = np.maximum(obs - sim, 0.0)
        under_sq = under[mask] ** 2 if np.any(mask) else under ** 2
        win = float(np.sqrt(np.mean(under_sq)))
    elif np.any(mask):
        win = float(np.sqrt(np.mean((sim[mask] - obs[mask]) ** 2)))
    else:
        win = overall

    if peak_excess:
        win = float(max(win - overall, 0.0))
    return float(overall_w) * overall + float(win_w) * win


LOSS_FUNCTIONS = {
    'RMSE': rmse_loss,
    'NSE': nse_loss,
    'PF_PEAK80': pf_peak80_loss,
}
