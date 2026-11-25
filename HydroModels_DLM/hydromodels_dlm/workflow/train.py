import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from hydromodels_dlm.config.model_config import PBM_PARAM_BOUNDS
from hydromodels_dlm.dataset.data_loader import (
    FloodEventDataLoader,
    get_input_layout,
    train_window_lookup,
    train_window_slice,
)
from hydromodels_dlm.dataset.sampler import build_period_event_packs
from hydromodels_dlm.dataset.data_source import group_file_ids_by_basin
from hydromodels_dlm.model.model_dict import MODEL_DICT
from hydromodels_dlm.model.physics_guided import (
    BasinEventSplitPhysicsModel,
    BasinStaticPhysicsModel,
    EventStaticPhysicsModel,
    PhysicsGuidedModel,
    apply_cemaneige_climatology,
    apply_pbm_grad_steps,
    driver_indices,
    event_head_output_size,
    is_physics_model,
    make_physics_model,
    pbm_grad_steps,
)
from hydromodels_dlm.utils.device import resolve_device
from hydromodels_dlm.utils.logging import (
    get_logger,
    log_detail,
    log_epoch_progress,
    log_section,
    manual_progress_bar,
    progress_iter,
)
from hydromodels_dlm.utils.seed import set_random_seed
from hydromodels_dlm.workflow.artifacts import BEST_CHECKPOINT, NORMALIZATION_SCALER
from hydromodels_dlm.workflow.losses import LOSS_FUNCTIONS
from hydromodels_dlm.workflow.transfer import load_init_weights


MODEL_HYPERPARAM_KEYS = (
    'input_size',
    'output_size',
    'hidden_size',
    'dropout',
    'input_proj',
)


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


def read_model_hyperparam(cfg):
    return dict((cfg.get('model_cfgs') or {}).get('model_hyperparam') or {})


def expected_output_size(cfg):
    name = str(cfg['model_cfgs']['model_name'])
    param_mode = str((cfg.get('training_cfgs') or {}).get('param_mode') or 'window')
    if is_physics_model(name):
        if param_mode == 'basin_event_split':
            return event_head_output_size(name)
        # window / event_static / basin_static all expose full PBM width
        # (basin_static ignores the LSTM head but keeps size for config checks)
        return len(PBM_PARAM_BOUNDS[name])
    outputs = (cfg.get('data_cfgs') or {}).get('variables', {}).get('dynamic_outputs') or {}
    if not outputs:
        raise ValueError('data_cfgs.variables.dynamic_outputs required to validate output_size')
    return len(outputs)


def validate_model_hyperparam(cfg, n_inputs):
    raw = read_model_hyperparam(cfg)
    unknown = set(raw) - set(MODEL_HYPERPARAM_KEYS)
    if unknown:
        raise ValueError(
            f'unknown model_hyperparam keys {sorted(unknown)!r}; allowed: {MODEL_HYPERPARAM_KEYS}'
        )
    hp = {k: raw[k] for k in MODEL_HYPERPARAM_KEYS if k in raw}

    for key in ('input_size', 'output_size'):
        if key not in hp or hp[key] is None:
            raise ValueError(
                f'model_hyperparam.{key} must be set explicitly in config (no default)'
            )

    input_size = int(hp['input_size'])
    output_size = int(hp['output_size'])
    if input_size != int(n_inputs):
        raise ValueError(
            f'model_hyperparam.input_size={input_size} != data features {n_inputs}'
        )

    expected_out = expected_output_size(cfg)
    if output_size != expected_out:
        raise ValueError(
            f'model_hyperparam.output_size={output_size} != expected {expected_out} '
            f"for model {cfg['model_cfgs']['model_name']!r}"
        )

    return {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_size': hp.get('hidden_size', 32),
        'dropout': hp.get('dropout', 0.0),
        'input_proj': hp.get('input_proj'),
    }


def build_model(cfg, *, n_inputs, warmup_length=365, basin_ids=None):
    name = cfg['model_cfgs']['model_name']
    entry = MODEL_DICT.get(name)
    if entry is None:
        raise ValueError(f'unknown model {name!r}; registered: {sorted(MODEL_DICT)}')

    hp = validate_model_hyperparam(cfg, n_inputs)
    param_mode = str((cfg.get('training_cfgs') or {}).get('param_mode') or 'window')
    if is_physics_model(name):
        pb_core = entry(warmup_length=warmup_length)
        if param_mode == 'basin_event_split':
            if not basin_ids:
                raise ValueError('basin_event_split requires basin_ids when building model')
            basin_ids = [str(b) for b in basin_ids]
            return BasinEventSplitPhysicsModel(
                n_lstm_inputs=hp['input_size'],
                pb_core=pb_core,
                n_basins=len(basin_ids),
                basin_ids=basin_ids,
                hidden_size=hp['hidden_size'],
                dropout=hp['dropout'],
                input_proj=hp['input_proj'],
            )
        if param_mode == 'basin_static':
            return BasinStaticPhysicsModel(pb_core)
        if param_mode == 'event_static':
            return EventStaticPhysicsModel(
                n_lstm_inputs=hp['input_size'],
                pb_core=pb_core,
                hidden_size=hp['hidden_size'],
                dropout=hp['dropout'],
                input_proj=hp['input_proj'],
            )
        if param_mode not in ('window', '', 'None', 'none'):
            raise ValueError(
                f"unknown param_mode {param_mode!r}; "
                "expected window|event_static|basin_static|basin_event_split"
            )
        return make_physics_model(
            n_lstm_inputs=hp['input_size'],
            output_size=hp['output_size'],
            hidden_size=hp['hidden_size'],
            dropout=hp['dropout'],
            input_proj=hp['input_proj'],
            pb_core=pb_core,
        )

    return entry(
        input_size=hp['input_size'],
        output_size=hp['output_size'],
        hidden_size=hp['hidden_size'],
        dropout=hp['dropout'],
        input_proj=hp['input_proj'],
    )


# ---------------------------------------------------------------------------
# Optimizer and criterion
# ---------------------------------------------------------------------------


def epoch_learning_rate(base_lr, decay, epoch):
    return float(base_lr) * (float(decay) ** (int(epoch) - 1))


def set_optimizer_learning_rate(optimizer, lr):
    value = float(lr)
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def build_optimizer(model, cfg):
    name = str(cfg['training_cfgs']['optimizer_name'])
    if name != 'Adam':
        raise ValueError(f'unsupported optimizer {name!r}')
    return optim.Adam(model.parameters(), lr=float(cfg['training_cfgs']['learning_rate']))


def build_criterion(cfg):
    import inspect

    name = str(cfg['training_cfgs']['loss_function']).upper()
    factory = LOSS_FUNCTIONS[name]
    loss_params = dict(cfg['training_cfgs'].get('loss_params') or {})

    if inspect.isclass(factory):
        if not loss_params:
            return factory()
        sig = inspect.signature(factory.__init__)
        allowed = {
            k: v for k, v in loss_params.items() if k in sig.parameters and k != 'self'
        }
        return factory(**allowed) if allowed else factory()

    criterion = factory()
    for key, value in loss_params.items():
        if hasattr(criterion, key):
            setattr(criterion, key, value)
    return criterion


# ---------------------------------------------------------------------------
# Batch forward
# ---------------------------------------------------------------------------


def model_forward(model, batch, warmup):
    if isinstance(model, PhysicsGuidedModel):
        drivers, lstm_inputs, _y, _basin_idx = batch
        return model(drivers, lstm_inputs)
    x, _y = batch
    return model(x)[:, warmup:, :]


def forward_batch(model, batch, device, warmup, *, physics=False):
    if physics:
        drivers_b, lstm_b, yb, basin_idx = batch
        drivers_b = drivers_b.to(device, non_blocking=True)
        lstm_b = lstm_b.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        basin_idx = basin_idx.to(device, non_blocking=True)
        drivers = drivers_b.transpose(0, 1)
        pred = model_forward(
            model,
            (drivers, lstm_b, yb, basin_idx),
            warmup,
        ).transpose(0, 1)
        return pred, yb

    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    pred = model_forward(model, (xb, yb), warmup)
    return pred, yb


def optimizer_step(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------


def format_rate(value):
    return f'{float(value):g}'


def log_gpu_memory(log, device, label):
    if device.type != 'cuda':
        return
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    log_detail(
        log,
        'GPU memory [%s]: peak allocated %.2f GiB, reserved %.2f GiB',
        label,
        alloc,
        reserved,
    )


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    warmup,
    *,
    desc='Train',
    show_bar=True,
    bar_leave=True,
    physics=False,
):
    model.train()
    total = 0.0
    n_batches = 0
    batch_iter = progress_iter(
        loader,
        desc=desc,
        enabled=show_bar,
        unit='batch',
        leave=bar_leave,
    )
    for batch in batch_iter:
        pred, yb = forward_batch(model, batch, device, warmup, physics=physics)
        loss = criterion(pred, yb)
        optimizer_step(loss, model, optimizer)
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------


def logical_basin_ids(loader, basin_ids):
    if isinstance(loader, FloodEventDataLoader):
        return sorted(group_file_ids_by_basin(basin_ids).keys())
    return sorted(str(b) for b in basin_ids)


def count_sliding_valid_batches(
    loader,
    basin_ids,
    scalers,
    *,
    period,
    batch_size,
    num_workers,
    model_name=None,
):
    period_key = str(period).lower()
    total = 0
    for basin_id in logical_basin_ids(loader, basin_ids):
        valid_loader = loader.build_period_loader(
            [basin_id],
            scalers,
            period=period_key,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            drop_last=False,
            model_name=model_name,
        )
        total += len(valid_loader)
    return total


def basin_sliding_valid_loss(
    model,
    loader,
    basin_id,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    model_name=None,
    physics=False,
    progress=None,
):
    period_key = str(period).lower()
    sample_warmup = 0 if physics else int(loader.warmup_length)
    valid_loader = loader.build_period_loader(
        [basin_id],
        scalers,
        period=period_key,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        model_name=model_name,
        pin_memory=device.type == 'cuda',
    )
    if len(valid_loader.dataset) == 0:
        raise ValueError(
            f'validation sampler has zero samples for basin {basin_id!r} '
            f'period {period_key!r}'
        )

    pred_parts = []
    target_parts = []
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            pred, yb = forward_batch(
                model,
                batch,
                device,
                sample_warmup,
                physics=physics,
            )
            pred_parts.append(pred.reshape(-1, pred.shape[-1]))
            target_parts.append(yb.reshape(-1, yb.shape[-1]))
    pred_all = torch.cat(pred_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    if progress is not None:
        progress.update(len(valid_loader))
    return float(criterion(pred_all, target_all).detach())


def basin_equal_sliding_valid_loss(
    model,
    loader,
    basin_ids,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    model_name=None,
    physics=False,
    show_bar=False,
    desc='Valid',
    bar_leave=False,
):
    basin_list = logical_basin_ids(loader, basin_ids)
    progress = None
    if show_bar:
        progress = manual_progress_bar(
            total=count_sliding_valid_batches(
                loader,
                basin_ids,
                scalers,
                period=period,
                batch_size=batch_size,
                num_workers=num_workers,
                model_name=model_name,
            ),
            desc=desc,
            unit='batch',
            leave=bar_leave,
        )
    try:
        losses = [
            basin_sliding_valid_loss(
                model,
                loader,
                basin_id,
                period,
                scalers,
                criterion,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                model_name=model_name,
                physics=physics,
                progress=progress,
            )
            for basin_id in basin_list
        ]
    finally:
        if progress is not None:
            progress.close()
    if not losses:
        raise ValueError(f'no validation samples for period {period!r}')
    return sum(losses) / len(losses)


def valid_loss(
    model,
    loader,
    basin_ids,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    desc='Valid',
    show_bar=True,
    bar_leave=True,
    model_name=None,
):
    physics = bool(model_name and is_physics_model(model_name))
    return basin_equal_sliding_valid_loss(
        model,
        loader,
        basin_ids,
        period,
        scalers,
        criterion,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
        physics=physics,
        show_bar=show_bar,
        desc=desc,
        bar_leave=bar_leave,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def log_training_setup(log, cfg, train_sampler, physics):
    tc = cfg['training_cfgs']
    warmup = int(train_sampler.warmup)
    horizon = int(train_sampler.horizon)
    base_lr = float(tc['learning_rate'])
    lr_decay = float(tc.get('learning_rate_decay', 1.0))
    log_section(log, '▶ Training')
    log_detail(
        log,
        'Optimizer: %s  |  Loss: %s  |  Learning rate: %s (decay: %s)',
        tc.get('optimizer_name', 'Adam'),
        tc.get('loss_function', 'RMSE'),
        format_rate(base_lr),
        format_rate(lr_decay),
    )
    log_detail(
        log,
        'Batch_Size: %d  |  Train samples: %d  |  sequence=warmup+forecast: %d+%d  |',
        int(tc['batch_size']),
        len(train_sampler),
        warmup,
        horizon,
    )
    log_detail(
        log,
        'Epochs: %d  |  Patience: %d',
        int(tc['epochs']),
        int(tc['patience']),
    )
    if physics:
        steps = pbm_grad_steps(cfg, horizon)
        log_detail(
            log,
            'PBM grad_steps: %s (forecast_length=%d)',
            'full' if steps <= 0 else steps,
            int(horizon),
        )


def train_model(
    cfg,
    loader,
    basin_ids,
    out_dir,
    device=None,
    *,
    init_weight_path=None,
    scalers=None,
):
    log = get_logger('train')
    tc = cfg['training_cfgs']
    set_random_seed(tc['random_seed'])
    device = device or resolve_device(tc.get('device'))
    model_name = cfg['model_cfgs']['model_name']
    physics = is_physics_model(model_name)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scalers is None:
        scalers = loader.fit_scalers(basin_ids)
    scaler_path = out_dir / NORMALIZATION_SCALER
    loader.save_scalers(scaler_path, scalers)

    event_agg = parse_event_loss_agg(tc.get('event_loss_agg'))
    batch_size = int(tc['batch_size'])
    num_workers = int(tc.get('num_workers', 0))
    pin_memory = device.type == 'cuda'

    # Event-stitched fl=1 path: train/valid share MEAN or LENGTH aggregation.
    # Supports SeqRegLSTM and physics-guided dXAJ (physical-Q targets).
    if event_agg is not None:
        if int(loader.forecast_length) != 1:
            raise ValueError(
                'event_loss_agg requires forecast_length=1 '
                f'(got {loader.forecast_length})'
            )

        n_inputs = loader.model_input_size(scalers)
        model = build_model(
            cfg,
            n_inputs=n_inputs,
            warmup_length=loader.warmup_length,
            basin_ids=basin_ids,
        ).to(device)
        if physics:
            apply_pbm_grad_steps(model, cfg, loader.forecast_length)
            apply_cemaneige_climatology(model, loader, basin_ids)
        if init_weight_path:
            load_init_weights(model, init_weight_path, loader, scalers, device)
            log_detail(log, 'Init weights: %s', init_weight_path)
        optimizer = build_optimizer(model, cfg)
        loss_name = str(tc['loss_function']).upper()
        loss_params = loss_params_from_cfg(cfg)
        event_bs = int(tc.get('event_batch_size', 8))
        grad_steps = pbm_grad_steps(cfg, loader.forecast_length)

        di = (
            driver_indices(model_name, loader.input_keys) if physics else None
        )
        train_packs = build_event_packs_for_period(
            loader, basin_ids, scalers, 'train', driver_indices=di
        )
        valid_packs = build_event_packs_for_period(
            loader, basin_ids, scalers, 'valid', driver_indices=di
        )
        if not train_packs:
            raise ValueError('training event packs empty')

        scale_mode = parse_scale_mode(loss_params)
        loss_space = parse_loss_space(loss_params, physics=physics)
        q_min = float(loss_params.get('q_min', 1e-3))
        # Peak scales follow the space the loss uses.
        scale_physics = physics and loss_space == 'physical'
        basin_scales = None
        if scale_mode == 'basin_median_peak':
            basin_scales = basin_median_peak_scales(
                train_packs, physics=scale_physics, q_min=q_min
            )

        epochs = int(tc['epochs'])
        patience = int(tc['patience'])
        best_loss = float('inf')
        best_epoch = 0
        stale = 0

        log_detail(
            log,
            'Event-stitched train/valid | model=%s | loss=%s | agg=%s | '
            'loss_space=%s | scale_mode=%s | peak_soft=%s | soft_temp=%s | '
            'train_events=%d | valid_events=%d | event_batch=%d | '
            'window_batch=%d | G=%s',
            model_name,
            loss_name,
            event_agg,
            loss_space,
            scale_mode,
            bool(loss_params.get('peak_soft', False)),
            loss_params.get('peak_soft_temp', 0.05),
            len(train_packs),
            len(valid_packs),
            event_bs,
            batch_size,
            grad_steps,
        )
        if basin_scales:
            vals = sorted(float(v) for v in basin_scales.values())
            mid = vals[len(vals) // 2]
            log_detail(
                log,
                'Basin median-peak scales (train): n=%d  median=%.4g  '
                'min=%.4g  max=%.4g',
                len(vals),
                mid,
                vals[0],
                vals[-1],
            )
        log_detail(
            log,
            'Optimizer: %s  |  Loss: %s  |  Learning rate: %s (decay: %s)',
            tc.get('optimizer_name', 'Adam'),
            loss_name,
            format_rate(tc['learning_rate']),
            format_rate(tc.get('learning_rate_decay', 1.0)),
        )
        log_detail(
            log,
            'Epochs: %d  |  Patience: %d  |  eval_mode=%s',
            epochs,
            patience,
            cfg['evaluation_cfgs'].get('eval_mode', 'sliding'),
        )

        for epoch in range(1, epochs + 1):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            lr = set_optimizer_learning_rate(
                optimizer,
                epoch_learning_rate(
                    float(tc['learning_rate']),
                    float(tc.get('learning_rate_decay', 1.0)),
                    epoch,
                ),
            )
            t0 = time.perf_counter()
            tr_loss, tr_steps, tr_n = period_event_objective(
                model,
                train_packs,
                device=device,
                warmup=int(loader.warmup_length),
                horizon=1,
                loss_name=loss_name,
                loss_params=loss_params,
                event_loss_agg=event_agg,
                window_batch_size=batch_size,
                train=True,
                optimizer=optimizer,
                event_batch_size=event_bs,
                show_bar=True,
                desc=f'Epoch {epoch}/{epochs} Train',
                physics=physics,
                basin_scales=basin_scales,
                scalers=scalers,
            )
            log_epoch_progress(
                log, epoch, epochs, 'train', max(tr_steps, 1),
                time.perf_counter() - t0,
            )
            if epoch == 1:
                log_gpu_memory(log, device, 'after epoch 1 train')
            t0 = time.perf_counter()
            va_loss, _, va_n = period_event_objective(
                model,
                valid_packs,
                device=device,
                warmup=int(loader.warmup_length),
                horizon=1,
                loss_name=loss_name,
                loss_params=loss_params,
                event_loss_agg=event_agg,
                window_batch_size=batch_size,
                train=False,
                optimizer=None,
                event_batch_size=event_bs,
                show_bar=True,
                desc=f'Epoch {epoch}/{epochs} Valid',
                physics=physics,
                basin_scales=basin_scales,
                scalers=scalers,
            )
            log_epoch_progress(
                log, epoch, epochs, 'valid', max(va_n, 1),
                time.perf_counter() - t0,
            )
            log_detail(
                log,
                'Epoch %d/%d Lr=%s  |  Train Loss=%.4f (events=%d)  |  '
                'Valid Loss=%.4f (events=%d)  |  agg=%s',
                epoch,
                epochs,
                format_rate(lr),
                tr_loss,
                tr_n,
                va_loss,
                va_n,
                event_agg,
            )
            if va_loss < best_loss:
                stale = 0
                best_epoch = epoch
                best_loss = va_loss
                ckpt = {
                    'model_state': model.state_dict(),
                    'model_name': model_name,
                    'n_inputs': n_inputs,
                    'model_hyperparam': validate_model_hyperparam(cfg, n_inputs),
                    'event_loss_agg': event_agg,
                    'loss_function': loss_name,
                    'basin_ids': [str(b) for b in basin_ids],
                    'param_mode': str(tc.get('param_mode') or 'window'),
                }
                torch.save(ckpt, out_dir / BEST_CHECKPOINT)
                log_detail(log, 'Model Update')
            else:
                stale += 1
                log_detail(
                    log, 'Epochs without Model Update: %d/%d', stale, patience
                )
                if stale >= patience:
                    log_detail(log, 'Early stop at epoch %d', epoch)
                    break

        return {
            'out_dir': str(out_dir),
            'best_valid_loss': best_loss,
            'best_epoch': best_epoch,
            'epochs_run': epoch,
            'scaler_path': str(scaler_path),
        }

    train_sampler = loader.build_train_sampler(
        basin_ids, scalers, model_name=model_name
    )
    if len(train_sampler) == 0:
        raise ValueError('training sampler has zero samples')

    batch_size = int(tc['batch_size'])
    num_workers = int(tc.get('num_workers', 0))
    pin_memory = device.type == 'cuda'
    train_loader = loader.build_train_loader(
        basin_ids,
        scalers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_sampler) > batch_size,
        model_name=model_name,
        pin_memory=pin_memory,
    )

    n_inputs = loader.model_input_size(scalers)
    model = build_model(
        cfg,
        n_inputs=n_inputs,
        warmup_length=loader.warmup_length,
        basin_ids=basin_ids,
    ).to(device)
    if physics:
        apply_pbm_grad_steps(model, cfg, loader.forecast_length)
        apply_cemaneige_climatology(model, loader, basin_ids)
    if init_weight_path:
        load_init_weights(model, init_weight_path, loader, scalers, device)
        log_detail(log, 'Init weights: %s', init_weight_path)
    optimizer = build_optimizer(model, cfg)
    criterion = build_criterion(cfg)
    sample_warmup = 0 if physics else int(loader.warmup_length)

    epochs = int(tc['epochs'])
    patience = int(tc['patience'])
    best_loss = float('inf')
    best_epoch = 0
    stale = 0

    log_training_setup(log, cfg, train_sampler, physics)

    for epoch in range(1, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr = set_optimizer_learning_rate(
            optimizer,
            epoch_learning_rate(
                float(tc['learning_rate']),
                float(tc.get('learning_rate_decay', 1.0)),
                epoch,
            ),
        )
        t0 = time.perf_counter()
        tr_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            sample_warmup,
            desc=f'Epoch {epoch}/{epochs} Train',
            show_bar=True,
            bar_leave=False,
            physics=physics,
        )
        log_epoch_progress(
            log, epoch, epochs, 'train', len(train_loader),
            time.perf_counter() - t0,
        )
        if epoch == 1:
            log_gpu_memory(log, device, 'after epoch 1 train')
        va_steps = count_sliding_valid_batches(
            loader,
            basin_ids,
            scalers,
            period='valid',
            batch_size=batch_size,
            num_workers=num_workers,
            model_name=model_name,
        )
        t0 = time.perf_counter()
        va_loss = valid_loss(
            model,
            loader,
            basin_ids,
            'valid',
            scalers,
            criterion,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            desc=f'Epoch {epoch}/{epochs} Valid',
            show_bar=True,
            bar_leave=False,
            model_name=model_name,
        )
        log_epoch_progress(
            log, epoch, epochs, 'valid', va_steps,
            time.perf_counter() - t0,
        )
        improved = va_loss < best_loss
        log_detail(
            log,
            'Epoch %d/%d Lr=%s  |  Train Loss=%.4f  |  Valid Loss=%.4f',
            epoch,
            epochs,
            format_rate(lr),
            tr_loss,
            va_loss,
        )

        if improved:
            stale = 0
            best_epoch = epoch
            best_loss = va_loss
            ckpt = {
                'model_state': model.state_dict(),
                'model_name': model_name,
                'n_inputs': n_inputs,
                'model_hyperparam': validate_model_hyperparam(cfg, n_inputs),
            }
            torch.save(ckpt, out_dir / BEST_CHECKPOINT)
            log_detail(log, 'Model Update')
        else:
            stale += 1
            log_detail(log, 'Epochs without Model Update: %d/%d', stale, patience)
            if stale >= patience:
                log_detail(log, 'Early stop at epoch %d', epoch)
                break

    return {
        'out_dir': str(out_dir),
        'best_valid_loss': best_loss,
        'best_epoch': best_epoch,
        'epochs_run': epoch,
        'scaler_path': str(scaler_path),
    }

# ---------------------------------------------------------------------------
# Event-stitched fl=1 training
# ---------------------------------------------------------------------------

EVENT_LOSS_AGGS = ('MEAN', 'LENGTH')


def parse_event_loss_agg(raw):
    if raw is None or str(raw).strip() == '':
        return None
    key = str(raw).upper()
    if key not in EVENT_LOSS_AGGS:
        raise ValueError(
            f'event_loss_agg must be one of {EVENT_LOSS_AGGS}, got {raw!r}'
        )
    return key


SCALE_MODES = ('none', 'event_peak', 'basin_median_peak')
LOSS_SPACES = ('auto', 'physical', 'normalized')


def parse_loss_space(loss_params, *, physics=False):
    """Where to evaluate the event loss.

    - ``auto``: physical Q for physics models, normalized Q for LSTM
    - ``physical``: raw mm (dXAJ default historically)
    - ``normalized``: basin ``y_scaler`` space (same as LSTM / PF-LSTM)

    For dXAJ + PF, ``normalized`` aligns the loss with the LSTM recipe while
    the model still predicts physical streamflow.
    """
    raw = dict(loss_params or {})
    mode = raw.get('loss_space', None)
    if mode is None or str(mode).strip() == '':
        return 'physical' if physics else 'normalized'
    key = str(mode).strip().lower()
    if key not in LOSS_SPACES:
        raise ValueError(f'loss_space must be one of {LOSS_SPACES}, got {mode!r}')
    if key == 'auto':
        return 'physical' if physics else 'normalized'
    return key


def transform_q_torch(q, y_scaler):
    """Differentiable basin Q normalization (matches ``FeatureScaler`` for Q)."""
    if len(y_scaler.methods) != 1:
        raise ValueError(
            f'transform_q_torch expects a single-feature y_scaler, got {y_scaler.methods}'
        )
    method = str(y_scaler.methods[0])
    mean = float(y_scaler.mean[0])
    std = float(y_scaler.std[0])
    std = std if abs(std) > 1e-8 else 1e-8
    if method == 'zscore':
        return (q - mean) / std
    if method == 'log1p_zscore':
        return (torch.log1p(q.clamp(min=0.0)) - mean) / std
    if method == 'prcp_log1p_zscore':
        prcp_scale = y_scaler.prcp_scale_at(0)
        if prcp_scale is None or float(prcp_scale) <= 0:
            raise ValueError('prcp_log1p_zscore requires positive prcp_scale')
        scaled = q.clamp(min=0.0) / float(prcp_scale)
        return (torch.log1p(scaled) - mean) / std
    raise ValueError(f'unsupported y_scaler method for torch transform: {method!r}')


def parse_scale_mode(loss_params):
    """Resolve loss scale mode from ``loss_params``.

    ``scale_mode``: none | event_peak | basin_median_peak.
    Unset / empty defaults to ``none`` (raw discharge units).
    """
    raw = dict(loss_params or {})
    mode = raw.get('scale_mode', None)
    if mode is None or str(mode).strip() == '':
        return 'none'
    key = str(mode).strip().lower()
    if key not in SCALE_MODES:
        raise ValueError(
            f'scale_mode must be one of {SCALE_MODES}, got {mode!r}'
        )
    return key


def event_obs_peak_np(y, flood_target=None):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if flood_target is not None:
        m = np.asarray(flood_target).reshape(-1)
        if m.shape[0] == y.shape[0]:
            m = m.astype(bool)
            if m.any():
                y = y[m]
    y = y[np.isfinite(y)]
    if y.size < 1:
        return float('nan')
    return float(np.max(y))


def basin_median_peak_scales(packs, *, physics=False, q_min=1e-3):
    """Per-basin scale = median of train-event peaks (target space).

    Physics uses ``y_raw`` (mm); LSTM uses ``y_norm``. Fixed from train packs
    and reused for valid/test objective scaling (no leakage).
    """
    peaks = {}
    for pack in packs:
        bid = str(pack['basin_id'])
        y = pack['y_raw'] if physics else pack['y_norm']
        peak = event_obs_peak_np(y, pack.get('flood_target'))
        if np.isfinite(peak):
            peaks.setdefault(bid, []).append(float(peak))
    scales = {}
    for bid, vals in peaks.items():
        scales[bid] = float(max(float(np.median(vals)), float(q_min)))
    return scales


def resolve_event_scale(
    pack,
    scale_mode,
    *,
    physics=False,
    basin_scales=None,
    q_min=1e-3,
):
    mode = scale_mode if scale_mode is not None else 'none'
    if mode == 'none':
        return None
    if mode == 'event_peak':
        y = pack['y_raw'] if physics else pack['y_norm']
        peak = event_obs_peak_np(y, pack.get('flood_target'))
        if not np.isfinite(peak):
            return float(q_min)
        return float(max(peak, float(q_min)))
    if mode == 'basin_median_peak':
        bid = str(pack['basin_id'])
        if not basin_scales or bid not in basin_scales:
            raise ValueError(
                f'basin_median_peak scale missing for basin_id={bid!r}'
            )
        return float(max(float(basin_scales[bid]), float(q_min)))
    raise ValueError(f'unknown scale_mode={mode!r}')


def series_rmse(sim, obs, eps=1e-8):
    finite = torch.isfinite(sim) & torch.isfinite(obs)
    n = int(finite.sum().item())
    if n < 1:
        return None, 0
    err = (sim[finite] - obs[finite]) ** 2
    return torch.sqrt(err.mean() + eps), n


def series_high_rmse(
    sim,
    obs,
    peak_ratio=0.8,
    eps=1e-8,
    *,
    soft=False,
    soft_temp=0.05,
    underonly=False,
):
    """HighRMSE near the event peak.

    Hard (default): steps with ``Qobs >= peak_ratio * Qp`` (same as eval).
    Soft: sigmoid weights in relative-peak space centered at ``peak_ratio``,
    temperature ``soft_temp`` (fraction of peak). Soft spreads peak gradients
    to near-peak hours instead of a hard 0/1 mask.

    ``underonly=True``: only count under-prediction error ``relu(Qobs−Qsim)``
    (flood peaks on RMSE-dXAJ are mostly underestimated; symmetric HighRMSE
    also punishes over-prediction and fights the XAJ constraint).
    """
    finite = torch.isfinite(sim) & torch.isfinite(obs)
    if int(finite.sum().item()) < 1:
        return None
    obs_f = obs[finite]
    sim_f = sim[finite]
    peak = torch.max(obs_f)
    if not torch.isfinite(peak):
        return None
    if underonly:
        sq = torch.relu(obs_f - sim_f) ** 2
    else:
        sq = (sim_f - obs_f) ** 2
    if not soft:
        high = obs_f >= float(peak_ratio) * peak
        if not bool(high.any()):
            return torch.sqrt(sq.mean() + eps)
        return torch.sqrt(sq[high].mean() + eps)

    # Soft high-flow weights: w = σ((Q/Qp − γ) / τ)
    peak_safe = peak.abs().clamp(min=eps)
    rel = obs_f / peak_safe
    tau = max(float(soft_temp), 1e-4)
    weights = torch.sigmoid((rel - float(peak_ratio)) / tau)
    denom = weights.sum().clamp(min=eps)
    if float(denom.detach()) < 1e-6:
        return torch.sqrt(sq.mean() + eps)
    return torch.sqrt((weights * sq).sum() / denom + eps)


def event_point_loss(
    sim,
    obs,
    *,
    loss_name,
    overall_weight=1.0,
    win_weight=0.5,
    peak_ratio=0.8,
    scale=None,
    q_min=1e-3,
    peak_soft=False,
    peak_soft_temp=0.05,
    peak_excess=False,
    peak_rel_excess=False,
    peak_underonly=False,
    eps=1e-8,
):
    """Loss on one stitched event series (1-D sim/obs). Returns (loss, n_valid).

    ``scale`` (optional): divide RMSE / HighRMSE by a positive constant
    (event peak or basin median peak) so physical-Q losses are comparable
    across basins/events. Default is raw discharge units (no scaling).

    ``peak_soft=True``: replace hard high-flow mask with sigmoid weights
    (see ``series_high_rmse``).

    ``peak_excess=True`` (PF only): ``L = RMSE + μ · relu(HighRMSE − RMSE)``
    so the peak term only penalizes *worse-than-overall* high-flow error.

    ``peak_rel_excess=True`` (PF only): dimensionally consistent relative form
    ``L = (RMSE / RMSE_det) + μ · relu(High / RMSE_det − 1)``.
    Both terms are dimensionless (forward value of the first term is 1; gradients
    still flow through RMSE). Overrides ``peak_excess``.

    ``peak_underonly=True``: HighRMSE uses only under-prediction errors.
    """
    name = str(loss_name).upper()

    rmse, n = series_rmse(sim, obs, eps=eps)
    if rmse is None:
        return None, 0

    scale_t = None
    if scale is not None:
        scale_t = torch.as_tensor(float(scale), dtype=rmse.dtype, device=rmse.device)
        rmse = rmse / scale_t.clamp(min=eps)
    if name == 'RMSE':
        return rmse, n
    if name == 'PF_PEAK80':
        high = series_high_rmse(
            sim,
            obs,
            peak_ratio=peak_ratio,
            eps=eps,
            soft=bool(peak_soft),
            soft_temp=float(peak_soft_temp),
            underonly=bool(peak_underonly),
        )
        if high is None:
            return float(overall_weight) * rmse, n
        if scale_t is not None:
            high = high / scale_t.clamp(min=eps)
        peak_term = high
        if peak_rel_excess:
            # Both terms dimensionless: L = RMSE/RMSE_det + μ·relu(High/RMSE_det − 1)
            rmse_det = rmse.detach().clamp(min=eps)
            peak_term = torch.relu(high / rmse_det - 1.0)
            return (
                float(overall_weight) * (rmse / rmse_det)
                + float(win_weight) * peak_term
            ), n
        if peak_excess:
            peak_term = torch.relu(high - rmse)
        return float(overall_weight) * rmse + float(win_weight) * peak_term, n
    raise ValueError(
        f'event-stitched training supports '
        f'RMSE/PF_PEAK80 only, got {loss_name!r}'
    )


def aggregate_event_losses(losses, lengths, mode):
    if not losses:
        raise ValueError('no event losses to aggregate')
    stacked = torch.stack(losses)
    if mode == 'MEAN':
        return stacked.mean()
    weights = torch.tensor(lengths, dtype=stacked.dtype, device=stacked.device)
    weights = weights / weights.sum().clamp(min=1.0)
    return (stacked * weights).sum()


def event_starts(x, y, flood, warmup, horizon):
    lookup = train_window_lookup(
        [x],
        [y],
        warmup=warmup,
        horizon=horizon,
        flood_target_mask=[flood],
    )
    # lookup entries are (0, start)
    return [int(start) for _, start in lookup]


def predict_event_stitched_fl1(
    model,
    x,
    y,
    flood,
    *,
    device,
    warmup,
    horizon=1,
    window_batch_size=512,
    drivers=None,
    basin_id=None,
):
    """Run fl=1 windows on one event; return stitched (sim, obs) on flood hours.

    SeqRegLSTM: ``x`` = LSTM inputs, ``y`` = normalized Q.
    Physics (dXAJ): pass raw ``drivers``; ``x`` = LSTM inputs; ``y`` = physical Q.
    BasinEventSplit: encode event params once (``basin_id`` required), freeze in-event.
    """
    if int(horizon) != 1:
        raise ValueError(
            f'event-stitched path requires forecast_length=1, got {horizon}'
        )
    starts = event_starts(x, y, flood, warmup, horizon)
    if not starts:
        return None, None

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(-1)
    drivers_t = None
    if drivers is not None:
        drivers_t = torch.from_numpy(np.asarray(drivers, dtype=np.float32))

    fixed_params = None
    if getattr(model, 'is_basin_event_split', False):
        if basin_id is None:
            raise ValueError('basin_event_split requires basin_id')
        if drivers_t is None:
            raise ValueError('basin_event_split requires drivers')
        x_full = x_t.unsqueeze(0).to(device, non_blocking=True)
        fixed_params = model.params_for_event(x_full, basin_id)
    elif getattr(model, 'is_event_static', False):
        if drivers_t is None:
            raise ValueError('event_static requires drivers')
        x_full = x_t.unsqueeze(0).to(device, non_blocking=True)
        fixed_params = model.params_for_event(x_full)
    elif getattr(model, 'is_basin_static', False):
        if drivers_t is None:
            raise ValueError('basin_static requires drivers')
        fixed_params = model.physical_params(batch_size=1)

    pred_chunks = []
    obs_chunks = []
    for i in range(0, len(starts), int(window_batch_size)):
        chunk = starts[i : i + int(window_batch_size)]
        xb = []
        yb = []
        db = []
        for start in chunk:
            input_sl, target_sl = train_window_slice(start, warmup, horizon)
            xb.append(x_t[input_sl])
            yb.append(y_t[target_sl])
            if drivers_t is not None:
                db.append(drivers_t[input_sl])
        xb = torch.stack(xb, dim=0).to(device, non_blocking=True)
        yb = torch.stack(yb, dim=0).to(device, non_blocking=True)
        if drivers_t is None:
            out = model(xb)[:, warmup:, :]
        elif fixed_params is not None:
            drivers_b = torch.stack(db, dim=0).to(device, non_blocking=True)
            drivers_tb = drivers_b.transpose(0, 1)
            n_win = drivers_tb.shape[1]
            params_b = fixed_params.expand(n_win, -1)
            out = model.forward_with_params(drivers_tb, params_b).transpose(0, 1)
        else:
            # PhysicsGuidedModel: drivers (T,B,D), lstm (B,T,F) → (horizon,B,1)
            drivers_b = torch.stack(db, dim=0).to(device, non_blocking=True)
            drivers_tb = drivers_b.transpose(0, 1)
            out = model(drivers_tb, xb).transpose(0, 1)
        pred_chunks.append(out.reshape(-1))
        obs_chunks.append(yb.reshape(-1))

    sim = torch.cat(pred_chunks, dim=0)
    obs = torch.cat(obs_chunks, dim=0)
    return sim, obs


def build_event_packs_for_period(
    loader,
    basin_ids,
    scalers,
    period,
    *,
    driver_indices=None,
):
    layout = get_input_layout(loader.input_layout_name)
    return build_period_event_packs(
        loader,
        [str(b) for b in basin_ids],
        scalers,
        layout,
        str(period).lower(),
        driver_indices=driver_indices,
    )


def loss_params_from_cfg(cfg):
    return dict(cfg['training_cfgs'].get('loss_params') or {})


def period_event_objective(
    model,
    packs,
    *,
    device,
    warmup,
    horizon,
    loss_name,
    loss_params,
    event_loss_agg,
    window_batch_size,
    train=False,
    optimizer=None,
    event_batch_size=8,
    show_bar=False,
    desc='EventLoss',
    physics=False,
    basin_scales=None,
    scalers=None,
):
    """Compute unified train/valid objective over event packs."""
    if not packs:
        raise ValueError('no event packs')
    agg = parse_event_loss_agg(event_loss_agg)
    q_min = float(loss_params.get('q_min', 1e-3))
    scale_mode = parse_scale_mode(loss_params)
    loss_space = parse_loss_space(loss_params, physics=physics)
    # Basin/event peak scales should match the space used by the loss.
    scale_physics = physics and loss_space == 'physical'
    params = {
        'overall_weight': float(loss_params.get('overall_weight', 1.0)),
        'win_weight': float(loss_params.get('win_weight', 0.5)),
        'peak_ratio': float(loss_params.get('peak_ratio', 0.8)),
        'q_min': q_min,
        'peak_soft': bool(loss_params.get('peak_soft', False)),
        'peak_soft_temp': float(loss_params.get('peak_soft_temp', 0.05)),
        'peak_excess': bool(loss_params.get('peak_excess', False)),
        'peak_rel_excess': bool(loss_params.get('peak_rel_excess', False)),
        'peak_underonly': bool(loss_params.get('peak_underonly', False)),
    }

    order = list(range(len(packs)))
    if train:
        random.shuffle(order)

    # Accumulate for epoch metric (all events).
    all_losses = []
    all_lengths = []

    # Mini-batches of events for optimizer steps (train only).
    def flush_batch(batch_losses, batch_lengths):
        nonlocal optimizer
        if not batch_losses:
            return
        loss = aggregate_event_losses(batch_losses, batch_lengths, agg)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # detach for metric accumulation
        return float(loss.detach())

    batch_losses = []
    batch_lengths = []
    steps = 0
    iterator = progress_iter(
        order,
        desc=desc,
        enabled=show_bar,
        unit='event',
        leave=False,
    )

    model.train(mode=train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx in iterator:
            pack = packs[idx]
            # Physics trains on physical Q; LSTM on normalized Q.
            y_target = pack['y_raw'] if physics else pack['y_norm']
            drivers = pack.get('drivers') if physics else None
            if physics and drivers is None:
                raise ValueError('physics event pack missing drivers')
            sim, obs = predict_event_stitched_fl1(
                model,
                pack['x'],
                y_target,
                pack['flood_target'],
                device=device,
                warmup=warmup,
                horizon=horizon,
                window_batch_size=window_batch_size,
                drivers=drivers,
                basin_id=pack.get('basin_id'),
            )
            if sim is None:
                continue
            if loss_space == 'normalized' and physics:
                if scalers is None:
                    raise ValueError(
                        'loss_space=normalized requires scalers for physics models'
                    )
                bid = str(pack['basin_id'])
                hub = scalers[bid]
                sim = transform_q_torch(sim, hub.y_scaler)
                obs = transform_q_torch(obs, hub.y_scaler)
            scale = resolve_event_scale(
                pack,
                scale_mode,
                physics=scale_physics,
                basin_scales=basin_scales,
                q_min=q_min,
            )
            loss_i, n_i = event_point_loss(
                sim,
                obs,
                loss_name=loss_name,
                scale=scale,
                **params,
            )
            if loss_i is None or n_i < 1:
                continue
            all_losses.append(loss_i.detach())
            all_lengths.append(n_i)
            if train:
                batch_losses.append(loss_i)
                batch_lengths.append(n_i)
                if len(batch_losses) >= int(event_batch_size):
                    flush_batch(batch_losses, batch_lengths)
                    steps += 1
                    batch_losses = []
                    batch_lengths = []

        if train and batch_losses:
            flush_batch(batch_losses, batch_lengths)
            steps += 1

    if not all_losses:
        raise ValueError('no valid event losses in period')
    # Epoch metric: full-period aggregate (same definition for train & valid).
    metric = aggregate_event_losses(all_losses, all_lengths, agg)
    return float(metric.detach()), steps, len(all_losses)
