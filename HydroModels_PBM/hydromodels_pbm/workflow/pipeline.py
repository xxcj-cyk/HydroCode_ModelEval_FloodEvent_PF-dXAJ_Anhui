import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from hydromodels_pbm.config.run_config import merge_config, prepare_config, save_basin_run_configs
from hydromodels_pbm.dataset.data_loader import (
    FloodEventDataLoader,
    LongTermDataLoader,
    create_data_loader,
)
from hydromodels_pbm.dataset.data_preprocess import log_flood_load, log_longterm_load
from hydromodels_pbm.dataset.data_source import (
    glob_basin_ids,
    input_data_root,
    load_basin_pair_list,
    parse_variables,
    resolve_basin_ids,
)
from hydromodels_pbm.utils.logging import (
    format_elapsed,
    get_logger,
    log_section,
    save_basin_log,
    setup_logging,
    start_basin_log,
    stop_basin_log,
)
from hydromodels_pbm.utils.normalization import normalize_params_dict
from hydromodels_pbm.utils.runtime_env import warmup_numba_runtime
from hydromodels_pbm.workflow.calibrate import calibrate_basin
from hydromodels_pbm.workflow.evaluate import (
    evaluate_basin,
    evaluation_periods,
    load_basin_params,
    read_params_payload,
    resolve_params_path,
    save_experiment_params_summary,
)
from hydromodels_pbm.workflow.metrics_artifacts import publish_experiment_period_metrics, sort_period_keys
from hydromodels_pbm.workflow.simulator import HydroSimulator, check_model_inputs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_basin_results(basin_id, cal_info, eval_info, metric_names, elapsed):
    log = get_logger('report')
    metrics = [str(m).upper() for m in metric_names]
    log_section(log, f'▶ Basin Results  [{basin_id}]')
    log.info('  Time %s', format_elapsed(elapsed))
    if cal_info:
        phys = cal_info.get('params_physical') or {}
        if phys:
            phys_fmt = {k: round(float(v), 3) for k, v in phys.items()}
            log.info('  Params: %s', json.dumps(phys_fmt, ensure_ascii=False))
        if cal_info.get('objective_value') is not None:
            log.info('  Calib objective: %.3f', cal_info['objective_value'])
    for period in sort_period_keys((eval_info or {}).get('metrics') or {}):
        row = eval_info['metrics'][period]
        parts = ', '.join(f'{name}={row[name]}' for name in metrics if name in row)
        log.info('  [%s] %s', period, parts)


# ---------------------------------------------------------------------------
# Basin run
# ---------------------------------------------------------------------------


def basin_file_ids(data_cfgs, basin_id):
    root = input_data_root(data_cfgs)
    bid = str(basin_id)
    if (root / f'{bid}.csv').is_file():
        return [bid]
    return glob_basin_ids(root, f'{bid}_*.csv')


def logical_run_ids(data_cfgs):
    probe = deepcopy(data_cfgs)
    probe['basin_ids'] = resolve_basin_ids(data_cfgs)
    return list(create_data_loader(probe).basin_ids)


def create_basin_loader(cfg, basin_id):
    data_cfgs = deepcopy(cfg['data_cfgs'])
    data_cfgs['basin_ids'] = basin_file_ids(data_cfgs, basin_id)
    loader = create_data_loader(data_cfgs)
    check_model_inputs(cfg['model_cfgs']['model_name'], loader.input_keys)
    return loader


def log_transplant_setup(loader, basin_id, cfg, params_json):
    setup_kw = {
        'model': cfg['model_cfgs']['model_name'],
        'algorithm': cfg['training_cfgs']['algorithm_name'],
        'output_dir': str(cfg['data_cfgs']['output_dir']),
        'input_path': str(resolve_params_path(params_json)),
    }
    if isinstance(loader, FloodEventDataLoader):
        log_flood_load(
            loader.root,
            basin_id,
            loader.list_period_event_ids(basin_id, 'calib'),
            loader.list_period_event_ids(basin_id, 'valid'),
            **setup_kw,
        )
    elif isinstance(loader, LongTermDataLoader):
        log_longterm_load(loader.root, basin_id, cfg['data_cfgs'], **setup_kw)
    else:
        raise TypeError(f'unsupported loader type: {type(loader)!r}')


def run_basin(basin_id, cfg, params_payload=None, init_norm=None):
    data_cfgs = cfg['data_cfgs']
    training_cfgs = cfg['training_cfgs']
    evaluation_cfgs = cfg['evaluation_cfgs']
    model_name = cfg['model_cfgs']['model_name']
    params_json = evaluation_cfgs.get('params_json')
    calib_dir = Path(data_cfgs['output_dir'])
    basin_key = str(basin_id)

    loader = create_basin_loader(cfg, basin_id)
    warmup_numba_runtime(
        model_name,
        data_cfgs['warmup_length'],
        training_cfgs['algorithm_params']['random_seed'],
    )

    start_basin_log()
    t0 = time.perf_counter()
    cal_info = None

    if params_json is None:
        cal_info = calibrate_basin(
            basin_id,
            calib_dir,
            model_name,
            training_cfgs['algorithm_name'],
            training_cfgs['objective_function'],
            training_cfgs['algorithm_params'],
            loader,
            init_norm=init_norm,
            event_loss_agg=training_cfgs.get('event_loss_agg', 'MEAN'),
            loss_params=training_cfgs.get('loss_params'),
        )
    else:
        log_transplant_setup(loader, basin_id, cfg, params_json)

    eval_info = evaluate_basin(
        basin_id,
        calib_dir,
        loader,
        evaluation_periods(),
        load_basin_params(calib_dir, basin_id, params_payload),
        evaluation_cfgs['metrics'],
        HydroSimulator(model_name),
    )
    loader.series_cache.clear()

    save_basin_run_configs(cfg, calib_dir, [basin_id])
    save_basin_log(calib_dir / basin_key / 'calibration_log.txt')
    stop_basin_log()

    return {
        'basin_id': basin_key,
        'cal_info': cal_info,
        'eval_info': eval_info,
        'elapsed': time.perf_counter() - t0,
    }


def run_basin_worker(job):
    setup_logging()
    return run_basin(
        job['basin_id'],
        job['cfg'],
        params_payload=job.get('params_payload'),
        init_norm=job.get('init_norm'),
    )


def record_basin_result(result, metric_names, cal_basins, eval_basins):
    basin_id = result['basin_id']
    if result['cal_info'] is not None:
        cal_basins[basin_id] = result['cal_info']
    eval_basins[basin_id] = result['eval_info']
    log_basin_results(
        basin_id,
        result['cal_info'],
        result['eval_info'],
        metric_names,
        result['elapsed'],
    )


def resolve_run_inputs(cfg):
    evaluation_cfgs = cfg['evaluation_cfgs']
    model_name = cfg['model_cfgs']['model_name']
    params_json = evaluation_cfgs.get('params_json')
    init_params_json = evaluation_cfgs.get('init_params_json')
    source_basin_id = evaluation_cfgs.get('source_basin_id')

    if params_json:
        return read_params_payload(params_json, source_basin_id), None
    if init_params_json:
        payload = read_params_payload(init_params_json, source_basin_id)
        return None, normalize_params_dict(payload['params'], model_name)
    return None, None


def run_basins(cfg, basin_ids, params_payload, init_norm):
    metric_names = list(cfg['evaluation_cfgs']['metrics'])
    cal_basins = {}
    eval_basins = {}
    job_kw = {'cfg': cfg, 'params_payload': params_payload, 'init_norm': init_norm}
    workers = min(cfg['training_cfgs']['max_workers'], len(basin_ids))

    if workers <= 1:
        for basin_id in basin_ids:
            record_basin_result(
                run_basin(basin_id, **job_kw),
                metric_names,
                cal_basins,
                eval_basins,
            )
        return cal_basins, eval_basins

    log = get_logger('pipeline')
    log_section(
        log,
        f'▶ Parallel Basins  [max_workers={workers}, basins={len(basin_ids)}]',
    )
    ctx = mp.get_context('spawn')
    jobs = [{'basin_id': basin_id, **job_kw} for basin_id in basin_ids]
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_basin_worker, job) for job in jobs]
        for future in as_completed(futures):
            record_basin_result(
                future.result(),
                metric_names,
                cal_basins,
                eval_basins,
            )
    return cal_basins, eval_basins


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_script_experiment(overrides, save_summary=True):
    cfg = merge_config(overrides)
    cfg = prepare_config(cfg, logical_run_ids(cfg['data_cfgs']))

    data_cfgs = cfg['data_cfgs']
    calib_dir = Path(data_cfgs['output_dir'])
    calib_dir.mkdir(parents=True, exist_ok=True)

    _, input_keys, _ = parse_variables(data_cfgs)
    check_model_inputs(cfg['model_cfgs']['model_name'], input_keys)

    params_payload, init_norm = resolve_run_inputs(cfg)
    basin_ids = list(data_cfgs['basin_ids'])
    cal_basins, eval_basins = run_basins(cfg, basin_ids, params_payload, init_norm)

    if save_summary:
        publish_experiment_period_metrics(calib_dir, basin_ids)
        save_experiment_params_summary(calib_dir, basin_ids)

    return {
        'status': 'completed',
        'calib_dir': str(calib_dir),
        'calibration': {'basins': cal_basins} if cal_basins else None,
        'evaluation': {'calib_dir': str(calib_dir), 'basins': eval_basins},
    }


# ---------------------------------------------------------------------------
# Pair-list runs
# ---------------------------------------------------------------------------


def save_pair_list_csv(output_dir, pairs, param_col, param_value):
    path = output_dir / (
        'param_transplant_list.csv'
        if param_col == 'params_json'
        else 'param_recalibrate_list.csv'
    )
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_basin_id', 'source_basin_id', param_col])
        for target_id, source_id in pairs:
            writer.writerow([target_id, source_id, param_value])


def run_pair_list_experiment(overrides):
    cfg = merge_config(overrides)
    evaluation_cfgs = cfg['evaluation_cfgs']
    params_json = evaluation_cfgs.get('params_json')
    init_params_json = evaluation_cfgs.get('init_params_json')
    transplant_list_csv = evaluation_cfgs.get('transplant_list_csv')

    if not transplant_list_csv:
        raise ValueError('pair-list run requires evaluation_cfgs.transplant_list_csv')
    if bool(params_json) == bool(init_params_json):
        raise ValueError(
            'pair-list run requires exactly one of params_json or init_params_json'
        )

    output_dir = Path(cfg['data_cfgs']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_basin_pair_list(transplant_list_csv)

    if params_json:
        save_pair_list_csv(output_dir, pairs, 'params_json', params_json)
    else:
        save_pair_list_csv(output_dir, pairs, 'init_params_json', init_params_json)

    for target_id, source_id in pairs:
        run_cfg = deepcopy(cfg)
        run_cfg['evaluation_cfgs']['source_basin_id'] = source_id
        run_cfg['data_cfgs']['basin_ids'] = glob_basin_ids(
            cfg['data_cfgs']['input_path'],
            f'{target_id}_*.csv',
        )
        run_script_experiment(run_cfg, save_summary=False)

    target_ids = [target_id for target_id, _ in pairs]
    publish_experiment_period_metrics(output_dir, target_ids)
    save_experiment_params_summary(output_dir, target_ids)

    return {
        'status': 'completed',
        'calib_dir': str(output_dir),
        'target_basin_ids': target_ids,
    }
