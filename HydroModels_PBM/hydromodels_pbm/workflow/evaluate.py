import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydromodels_pbm.dataset.data_loader import FloodEventDataLoader, LongTermDataLoader
from hydromodels_pbm.dataset.data_preprocess import (
    build_driver_arrays,
    check_arrays,
    clip_range,
    period_step_indices,
)
from hydromodels_pbm.dataset.data_source import format_timestamp
from hydromodels_pbm.utils.logging import write_json
from hydromodels_pbm.workflow.metrics import METRIC_REGISTRY
from hydromodels_pbm.workflow.metrics_artifacts import (
    aggregate_event_metrics,
    format_metric_value,
    write_basin_metrics,
    write_basin_metrics_longterm,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def resolve_params_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def read_params_payload(path, source_basin_id=None):
    path = resolve_params_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'missing params json: {path}')
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        if not source_basin_id:
            raise ValueError(
                f'{path} is a multi-basin params file; source_basin_id is required'
            )
        key = str(source_basin_id)
        for row in payload:
            if str(row.get('basin_id')) == key:
                if 'params' not in row:
                    raise KeyError(f'{path} row {key!r} must contain a params field')
                return row
        raise KeyError(f'missing basin {key!r} in {path}')
    if 'params' not in payload:
        raise KeyError(f'{path} must contain a params field')
    return payload


def load_basin_params(calib_dir, basin_id, params_payload=None):
    if params_payload is not None:
        basin_dir = Path(calib_dir) / str(basin_id)
        basin_dir.mkdir(parents=True, exist_ok=True)
        out = dict(params_payload)
        out['basin_id'] = str(basin_id)
        write_json(out, basin_dir / 'best_params.json')
        return dict(params_payload['params'])
    path = Path(calib_dir) / str(basin_id) / 'best_params.json'
    return dict(read_params_payload(path)['params'])


def write_best_params(basin_dir, basin_id, model_name, params):
    rounded = {key: round(float(value), 3) for key, value in params.items()}
    payload = {
        'basin_id': str(basin_id),
        'model_name': model_name,
        'params': rounded,
    }
    write_json(payload, Path(basin_dir) / 'best_params.json')
    return rounded


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def evaluation_periods():
    return ('calib', 'valid')


def prepare_basin_eval_dir(calib_dir, basin_id):
    basin_dir = Path(calib_dir) / str(basin_id)
    basin_dir.mkdir(parents=True, exist_ok=True)
    ts_path = basin_dir / 'timeseries.csv'
    if ts_path.is_file():
        ts_path.unlink()
    return basin_dir, ts_path


# ---------------------------------------------------------------------------
# Metrics and CSV
# ---------------------------------------------------------------------------


def compute_metrics(qsim, qobs, metric_names, area_km2=None):
    out = {}
    for name in metric_names:
        key = str(name).upper()
        fn = METRIC_REGISTRY[key]
        if key in (
            'RMSE_Q',
            'HIGHRMSE_Q',
            'HIGHRMSE70_Q',
            'HIGHRMSE80_Q',
            'HIGHRMSE90_Q',
        ):
            out[key] = float(fn(qsim, qobs, area_km2))
        else:
            out[key] = float(fn(qsim, qobs))
    return out


def basin_area_km2(loader, basin_id):
    paths = sorted(Path(loader.root).glob('Attributes_*.csv'))
    if not paths:
        return None
    df = pd.read_csv(paths[0])
    if 'Area' not in df.columns:
        return None
    id_col = 'Basin_ID' if 'Basin_ID' in df.columns else df.columns[0]
    keys = [str(basin_id)]
    parts = keys[0].split('_')
    if len(parts) >= 3:
        keys.append(f'{parts[0]}_{parts[1]}')
    for key in keys:
        hit = df.loc[df[id_col].astype(str) == key, 'Area']
        if not hit.empty:
            area = float(hit.iloc[0])
            if np.isfinite(area) and area > 0:
                return area
    return None


def save_event_metrics(basin_dir, rows, metric_names):
    cols = ['period', 'basin_id', 'event_id'] + [str(m).upper() for m in metric_names]
    pd.DataFrame(rows)[cols].to_csv(
        basin_dir / 'event_metrics.csv',
        index=False,
        encoding='utf-8-sig',
    )


def save_experiment_params_summary(calib_dir, basin_ids):
    calib_dir = Path(calib_dir)
    rows = []
    for basin_id in sorted(str(b) for b in basin_ids):
        path = calib_dir / basin_id / 'best_params.json'
        if not path.is_file():
            continue
        with open(path, encoding='utf-8') as f:
            rows.append(json.load(f))
    if rows:
        write_json(rows, calib_dir / 'best_params_all.json')


def append_timeseries_csv(
    path,
    times,
    period,
    p_and_e,
    qobs,
    qsim,
    input_keys,
    *,
    times_true=None,
    event_id=None,
):
    n = qsim.shape[0]
    data = {
        'time': pd.to_datetime(times).map(format_timestamp),
        'period': period,
    }
    cols = ['time', 'period']
    if times_true is not None:
        data['time_true'] = pd.to_datetime(times_true).map(format_timestamp)
        cols.append('time_true')
    if event_id is not None:
        data['event_id'] = event_id
        cols.append('event_id')
    for i, key in enumerate(input_keys):
        data[key] = p_and_e[:n, 0, i]
    data['Qobs'] = qobs[:n, 0, 0]
    data['Qsim'] = qsim[:n, 0, 0]
    cols.extend([*input_keys, 'Qobs', 'Qsim'])
    df = pd.DataFrame(data)[cols]
    df.to_csv(
        path,
        mode='a',
        header=not path.is_file(),
        index=False,
        encoding='utf-8-sig',
    )


def event_times_true(series, warmup, eval_idx):
    if 'time_true' not in series.columns:
        return None
    return series.iloc[warmup + eval_idx]['time_true'].to_numpy()


# ---------------------------------------------------------------------------
# Long-term data
# ---------------------------------------------------------------------------


def load_longterm_eval_data(loader, basin_id, span, warmup):
    series = clip_range(loader.read_series(basin_id), span)
    p_and_e, qobs = build_driver_arrays(series, loader.input_keys, loader.obs_key)
    check_arrays(
        p_and_e,
        qobs,
        loader.input_keys,
        name=f'{span[0]}_{span[1]}',
        warmup_length=warmup,
    )
    if len(series.index) != p_and_e.shape[0]:
        raise ValueError(
            f'time axis length {len(series.index)} != driver length {p_and_e.shape[0]}'
        )
    return series, p_and_e


# ---------------------------------------------------------------------------
# Basin evaluation
# ---------------------------------------------------------------------------


def evaluate_flood_event_basin(
    basin_id,
    calib_dir,
    loader,
    periods,
    params,
    metric_names,
    simulator,
):
    warmup = loader.warmup_length
    basin_dir, ts_path = prepare_basin_eval_dir(calib_dir, basin_id)
    area_km2 = basin_area_km2(loader, basin_id)

    period_metrics = {}
    event_rows = []
    for period in periods:
        event_metric_rows = []
        for ev in loader.load_period_events(basin_id, period):
            idx = np.asarray(ev['eval_idx'], dtype=int)
            qsim_p = simulator.simulate(
                ev['p_and_e'], params, warmup_length=warmup
            )[idx]
            qobs_p = ev['qobs'][warmup + idx]
            metrics = compute_metrics(
                qsim_p[:, 0, 0], qobs_p[:, 0, 0], metric_names, area_km2=area_km2
            )
            event_metric_rows.append(metrics)
            append_timeseries_csv(
                ts_path,
                ev['series'].index[warmup + idx],
                period,
                ev['p_and_e'][warmup + idx],
                qobs_p.reshape(-1, 1, 1),
                qsim_p,
                loader.input_keys,
                times_true=event_times_true(ev['series'], warmup, idx),
                event_id=ev['event_id'],
            )
            event_rows.append(
                {
                    'period': period,
                    'basin_id': str(basin_id),
                    'event_id': ev['event_id'],
                    **{
                        name: format_metric_value(name, value)
                        for name, value in metrics.items()
                    },
                }
            )
        if not event_metric_rows:
            continue
        period_metrics[period] = aggregate_event_metrics(
            event_metric_rows,
            metric_names,
        )

    metrics_by_period = write_basin_metrics(basin_dir, str(basin_id), period_metrics)
    save_event_metrics(basin_dir, event_rows, metric_names)
    return {'basin_id': str(basin_id), 'metrics': metrics_by_period}


def evaluate_longterm_basin(
    basin_id,
    calib_dir,
    loader,
    periods,
    params,
    metric_names,
    simulator,
):
    warmup = loader.warmup_length
    basin_dir, ts_path = prepare_basin_eval_dir(calib_dir, basin_id)

    period_ranges = {period: loader.date_range(period, basin_id) for period in periods}
    span = [
        min(date_range[0] for date_range in period_ranges.values()),
        max(date_range[1] for date_range in period_ranges.values()),
    ]
    series, p_and_e = load_longterm_eval_data(loader, basin_id, span, warmup)
    times = series.index

    qsim_full = simulator.simulate(p_and_e, params, warmup_length=warmup)
    sim_times = times[warmup : warmup + qsim_full.shape[0]]
    if len(sim_times) != qsim_full.shape[0]:
        raise ValueError('simulation output length does not match time axis')

    qobs_at_sim = (
        series.loc[sim_times, loader.obs_key]
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1, 1)
    )

    area_km2 = basin_area_km2(loader, basin_id)
    period_metrics = {}
    for period in periods:
        idx = period_step_indices(times, warmup, period_ranges[period])
        qsim_p = qsim_full[idx]
        qobs_p = qobs_at_sim[idx]
        period_metrics[period] = compute_metrics(
            qsim_p[:, 0, 0], qobs_p[:, 0, 0], metric_names, area_km2=area_km2
        )
        append_timeseries_csv(
            ts_path,
            sim_times[idx],
            period,
            p_and_e[warmup + idx],
            qobs_p,
            qsim_p,
            loader.input_keys,
        )

    metrics_by_period = write_basin_metrics_longterm(
        basin_dir, str(basin_id), period_metrics
    )
    return {'basin_id': str(basin_id), 'metrics': metrics_by_period}


def evaluate_basin(
    basin_id,
    calib_dir,
    loader,
    periods,
    params,
    metric_names,
    simulator,
):
    if isinstance(loader, FloodEventDataLoader):
        return evaluate_flood_event_basin(
            basin_id,
            calib_dir,
            loader,
            periods,
            params,
            metric_names,
            simulator,
        )
    if isinstance(loader, LongTermDataLoader):
        return evaluate_longterm_basin(
            basin_id,
            calib_dir,
            loader,
            periods,
            params,
            metric_names,
            simulator,
        )
    raise TypeError(f'unsupported loader type: {type(loader)!r}')
