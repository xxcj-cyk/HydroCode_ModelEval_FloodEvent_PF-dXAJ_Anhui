from functools import partial

import numpy as np


# ---------------------------------------------------------------------------
# Valid samples
# ---------------------------------------------------------------------------


def valid_obs_sim(observed, simulated, min_samples=1):
    obs = np.asarray(observed, dtype=np.float64).ravel()
    sim = np.asarray(simulated, dtype=np.float64).ravel()
    if obs.shape != sim.shape:
        raise ValueError(
            f'observed and simulated length mismatch: {obs.shape} vs {sim.shape}'
        )
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    if obs_v.size < min_samples:
        return None, None
    return obs_v, sim_v


def safe_ratio(numerator, denominator):
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    value = numerator / denominator
    return float(value) if np.isfinite(value) else np.nan


# ---------------------------------------------------------------------------
# Correlation and efficiency
# ---------------------------------------------------------------------------


def nash_sutcliffe_efficiency(simulated, observed):
    """Nash–Sutcliffe efficiency (NSE); same as DC (确定性系数) in GB/T practice."""
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    denom = np.sum((obs_v - np.mean(obs_v)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((sim_v - obs_v) ** 2) / denom)


def kling_gupta_efficiency(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    std_ratio = safe_ratio(np.std(sim_v), np.std(obs_v))
    mean_ratio = safe_ratio(np.mean(sim_v), np.mean(obs_v))
    corr_coef = np.corrcoef(obs_v, sim_v)[0, 1]
    if np.isnan(std_ratio) or np.isnan(mean_ratio) or np.isnan(corr_coef):
        return np.nan
    return float(
        1.0
        - np.sqrt(
            (corr_coef - 1) ** 2
            + (std_ratio - 1) ** 2
            + (mean_ratio - 1) ** 2
        )
    )


def pearson_correlation(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=2)
    if obs_v is None:
        return np.nan
    if np.std(obs_v) <= 1e-12 or np.std(sim_v) <= 1e-12:
        return np.nan
    corr = np.corrcoef(sim_v, obs_v)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def coefficient_of_determination(simulated, observed):
    corr = pearson_correlation(simulated, observed)
    return np.nan if np.isnan(corr) else float(corr * corr)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def root_mean_squared_error(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float(np.sqrt(np.mean((sim_v - obs_v) ** 2)))


def mean_bias_error(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float(np.mean(sim_v - obs_v))


def high_flow_root_mean_squared_error(simulated, observed, high_ratio=0.8):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    peak_observed = np.max(obs_v)
    if peak_observed <= 0 or not np.isfinite(peak_observed):
        return np.nan
    high_flow_mask = obs_v >= peak_observed * high_ratio
    if not np.any(high_flow_mask):
        return np.nan
    mean_squared_diff = np.mean((sim_v[high_flow_mask] - obs_v[high_flow_mask]) ** 2)
    if not np.isfinite(mean_squared_diff) or mean_squared_diff < 0:
        return np.nan
    return float(np.sqrt(mean_squared_diff))


def root_mean_squared_error_q(simulated, observed, area_km2=None):
    """RMSE in m³/s when Q is in mm: RMSE_mm * Area / 3.6 (hourly)."""
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return np.nan
    rmse_mm = root_mean_squared_error(simulated, observed)
    if not np.isfinite(rmse_mm):
        return np.nan
    return float(rmse_mm * float(area_km2) / 3.6)


def high_flow_root_mean_squared_error_q(simulated, observed, area_km2=None, high_ratio=0.8):
    """High-flow RMSE in m³/s when Q is in mm: HighRMSE_mm * Area / 3.6."""
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return np.nan
    high_mm = high_flow_root_mean_squared_error(
        simulated, observed, high_ratio=high_ratio
    )
    if not np.isfinite(high_mm):
        return np.nan
    return float(high_mm * float(area_km2) / 3.6)


# ---------------------------------------------------------------------------
# Peak and flow volume
# ---------------------------------------------------------------------------


def peak_flow_error(simulated, observed):
    """Relative peak error (%): 100 * (Qsim_peak - Qobs_peak) / Qobs_peak."""
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    ratio = safe_ratio(np.max(sim_v) - np.max(obs_v), np.max(obs_v))
    return np.nan if np.isnan(ratio) else float(ratio * 100)


def absolute_peak_flow_error(simulated, observed):
    """Absolute peak error (same units as Q): Qsim_peak - Qobs_peak."""
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float(np.max(sim_v) - np.max(obs_v))


def peak_flow_error_abs(simulated, observed):
    """Absolute relative peak error (%): |PFE|."""
    value = peak_flow_error(simulated, observed)
    return np.nan if np.isnan(value) else float(abs(value))


def peak_timing_error(simulated, observed, timestep=1.0):
    """Peak timing error (signed; multiply by timestep, default 1 step)."""
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    return float((np.argmax(sim_v) - np.argmax(obs_v)) * timestep)


def peak_timing_error_abs(simulated, observed, timestep=1.0):
    """Absolute peak timing error: |PTE|."""
    value = peak_timing_error(simulated, observed, timestep=timestep)
    return np.nan if np.isnan(value) else float(abs(value))


def volume_error(simulated, observed):
    """Relative event volume error (%): 100 * (ΣQsim - ΣQobs) / ΣQobs."""
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=1)
    if obs_v is None:
        return np.nan
    ratio = safe_ratio(np.sum(sim_v) - np.sum(obs_v), np.sum(obs_v))
    return np.nan if np.isnan(ratio) else float(ratio * 100)


def volume_error_abs(simulated, observed):
    """Absolute relative volume error (%): |VE|."""
    value = volume_error(simulated, observed)
    return np.nan if np.isnan(value) else float(abs(value))


# GB/T-style event pass/fail (1/0). Basin/regional aggregation uses the mean
# of these flags (= 合格率); never median — see aggregate_event_metrics.
PFE_QUAL_LIMIT = 20.0  # |PFE| <= 20%
VE_QUAL_LIMIT = 20.0  # |VE| <= 20%
PTE_QUAL_LIMIT = 2.0  # |PTE| <= 2 h (fixed for event simulation)
QUAL_RATE_METRICS = frozenset({'QR', 'QR_PFE', 'QR_VE', 'QR_PTE'})


def peak_qualified(simulated, observed):
    value = peak_flow_error_abs(simulated, observed)
    if not np.isfinite(value):
        return np.nan
    return 1.0 if value <= PFE_QUAL_LIMIT else 0.0


def volume_qualified(simulated, observed):
    value = volume_error_abs(simulated, observed)
    if not np.isfinite(value):
        return np.nan
    return 1.0 if value <= VE_QUAL_LIMIT else 0.0


def peak_timing_qualified(simulated, observed, timestep=1.0):
    value = peak_timing_error_abs(simulated, observed, timestep=timestep)
    if not np.isfinite(value):
        return np.nan
    return 1.0 if value <= PTE_QUAL_LIMIT else 0.0


def event_qualified(simulated, observed, timestep=1.0):
    """Combined: |PFE|<=20%, |VE|<=20%, |PTE|<=2 h."""
    flags = (
        peak_qualified(simulated, observed),
        volume_qualified(simulated, observed),
        peak_timing_qualified(simulated, observed, timestep=timestep),
    )
    if any(not np.isfinite(v) for v in flags):
        return np.nan
    return 1.0 if all(v >= 1.0 for v in flags) else 0.0


def flow_high_volume_bias(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=5)
    if obs_v is None:
        return np.nan
    sim_sorted = np.sort(sim_v)
    obs_sorted = np.sort(obs_v)
    start = min(len(sim_sorted) - 1, int(0.98 * len(sim_sorted)))
    numerator = np.sum(sim_sorted[start:] - obs_sorted[start:])
    denominator = np.sum(obs_sorted[start:])
    ratio = safe_ratio(numerator, denominator)
    return np.nan if np.isnan(ratio) else float(ratio * 100)


def flow_low_volume_bias(simulated, observed):
    obs_v, sim_v = valid_obs_sim(observed, simulated, min_samples=5)
    if obs_v is None:
        return np.nan
    sim_sorted = np.sort(sim_v)
    obs_sorted = np.sort(obs_v)
    stop = max(1, int(0.3 * len(sim_sorted)))
    numerator = np.sum(sim_sorted[:stop] - obs_sorted[:stop])
    denominator = np.sum(obs_sorted[:stop])
    ratio = safe_ratio(numerator, denominator)
    return np.nan if np.isnan(ratio) else float(ratio * 100)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


METRIC_REGISTRY = {
    'NSE': nash_sutcliffe_efficiency,
    'KGE': kling_gupta_efficiency,
    'CORR': pearson_correlation,
    'R2': coefficient_of_determination,
    'RMSE': root_mean_squared_error,
    'MBE': mean_bias_error,
    'HIGHRMSE': high_flow_root_mean_squared_error,
    'HIGHRMSE70': partial(high_flow_root_mean_squared_error, high_ratio=0.7),
    'HIGHRMSE80': high_flow_root_mean_squared_error,
    'HIGHRMSE90': partial(high_flow_root_mean_squared_error, high_ratio=0.9),
    'RMSE_Q': root_mean_squared_error_q,
    'HIGHRMSE_Q': high_flow_root_mean_squared_error_q,
    'HIGHRMSE70_Q': partial(high_flow_root_mean_squared_error_q, high_ratio=0.7),
    'HIGHRMSE80_Q': high_flow_root_mean_squared_error_q,
    'HIGHRMSE90_Q': partial(high_flow_root_mean_squared_error_q, high_ratio=0.9),
    'PFE': peak_flow_error,
    'APFE': absolute_peak_flow_error,
    'PFE_ABS': peak_flow_error_abs,
    'PTE': peak_timing_error,
    'PTE_ABS': peak_timing_error_abs,
    'VE': volume_error,
    'VE_ABS': volume_error_abs,
    'QR_PFE': peak_qualified,
    'QR_VE': volume_qualified,
    'QR_PTE': peak_timing_qualified,
    'QR': event_qualified,
    'FHV': flow_high_volume_bias,
    'FLV': flow_low_volume_bias,
}

# Same set as PBM Sec0 XAJ scripts (step1/step2).
FLOOD_EVENT_METRICS = [
    'NSE',
    'KGE',
    'RMSE',
    'HIGHRMSE',
    'HIGHRMSE70',
    'HIGHRMSE80',
    'HIGHRMSE90',
    'RMSE_Q',
    'HIGHRMSE_Q',
    'HIGHRMSE70_Q',
    'HIGHRMSE80_Q',
    'HIGHRMSE90_Q',
    'PFE',
    'PFE_ABS',
    'PTE',
    'PTE_ABS',
    'VE',
    'VE_ABS',
    'QR_PFE',
    'QR_VE',
    'QR_PTE',
    'QR',
]
