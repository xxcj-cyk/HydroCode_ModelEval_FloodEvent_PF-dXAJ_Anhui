import contextlib
import re
import sys
from functools import partial

import numpy as np
import spotpy
from spotpy.parameter import Uniform

from hydromodels_pbm.utils.logging import append_basin_log, get_logger, log_section
from hydromodels_pbm.utils.normalization import denormalize
from hydromodels_pbm.workflow.objectives import LOSS_FUNCTIONS
from hydromodels_pbm.workflow.simulator import HydroSimulator, check_model_inputs


def bind_loss_fn(objective_function, loss_params=None):
    name = str(objective_function).upper()
    if name not in LOSS_FUNCTIONS:
        raise KeyError(
            f'unknown objective_function {objective_function!r}; '
            f'choose from: {sorted(LOSS_FUNCTIONS)}'
        )
    fn = LOSS_FUNCTIONS[name]
    if not loss_params:
        return fn
    return partial(fn, **dict(loss_params))


# ---------------------------------------------------------------------------
# Event loss aggregation
#   MEAN   = equal weight per event
#   LENGTH = weight by event length (#timesteps)
# ---------------------------------------------------------------------------


def collect_event_losses(observed_list, simulated_list, loss_fn):
    if len(observed_list) != len(simulated_list):
        raise ValueError('observed_list and simulated_list length mismatch')
    values = []
    lengths = []
    for obs, sim in zip(observed_list, simulated_list):
        obs_arr = np.asarray(obs, dtype=np.float64).ravel()
        sim_arr = np.asarray(sim, dtype=np.float64).ravel()
        if obs_arr.size == 0:
            continue
        if obs_arr.shape != sim_arr.shape:
            raise ValueError(
                f'event obs/sim length mismatch: {obs_arr.shape} vs {sim_arr.shape}'
            )
        values.append(float(loss_fn(obs_arr, sim_arr)))
        lengths.append(float(obs_arr.size))
    if not values:
        raise ValueError('no event loss values')
    return values, lengths


def mean_event_loss(observed_list, simulated_list, loss_fn):
    values, _lengths = collect_event_losses(observed_list, simulated_list, loss_fn)
    return float(np.mean(values))


def length_event_loss(observed_list, simulated_list, loss_fn):
    values, lengths = collect_event_losses(observed_list, simulated_list, loss_fn)
    lengths = np.asarray(lengths, dtype=np.float64)
    weights = lengths / lengths.sum()
    return float(np.dot(weights, np.asarray(values, dtype=np.float64)))


EVENT_LOSS_AGGREGATORS = {
    'MEAN': mean_event_loss,
    'LENGTH': length_event_loss,
}


# ---------------------------------------------------------------------------
# SpotPy setup
# ---------------------------------------------------------------------------


class SpotPyCalibrateBase:
    def __init__(self, model_name, objective_function, input_keys, loss_params=None):
        check_model_inputs(model_name, input_keys)
        self.model_name = model_name
        self.objective_function = objective_function.upper()
        self.loss_params = dict(loss_params) if loss_params else {}
        self.loss_fn = bind_loss_fn(self.objective_function, self.loss_params)
        self.simulator = HydroSimulator(model_name)
        self.param_names = self.simulator.param_names
        self.spotpy_params = [
            Uniform(name, low=0.0, high=1.0) for name in self.param_names
        ]

    def parameters(self):
        return spotpy.parameter.generate(self.spotpy_params)

    def denormalize_params(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(1, -1)
        try:
            return denormalize(arr, self.model_name)
        except (ArithmeticError, ValueError, RuntimeError):
            return None

    def invalid_simulation(self, template):
        return np.full_like(template, np.nan, dtype=np.float64)

    def objectivefunction(self, simulation, evaluation, params=None):
        if np.isnan(simulation).all():
            return 1e12
        return float(self.loss_fn(evaluation, simulation))


class LongTermCalibrateSetup(SpotPyCalibrateBase):
    def __init__(
        self,
        p_and_e,
        qobs,
        warmup_length,
        calib_idx,
        model_name,
        objective_function,
        input_keys,
        loss_params=None,
    ):
        super().__init__(
            model_name, objective_function, input_keys, loss_params=loss_params
        )
        self.calib_idx = np.asarray(calib_idx, dtype=int)
        if self.calib_idx.size == 0:
            raise ValueError('calib period has no steps after warmup')
        self.p_and_e = p_and_e
        self.warmup_length = warmup_length
        self.true_obs = qobs[warmup_length + self.calib_idx, :, :]

    def simulation(self, x):
        physical = self.denormalize_params(x)
        if physical is None:
            return self.invalid_simulation(self.true_obs)
        qsim = self.simulator.simulate(
            self.p_and_e,
            physical,
            warmup_length=self.warmup_length,
        )
        return qsim[self.calib_idx]

    def evaluation(self):
        return self.true_obs


class FloodEventCalibrateSetup(SpotPyCalibrateBase):
    def __init__(
        self,
        events,
        warmup_length,
        model_name,
        objective_function,
        input_keys,
        event_loss_agg='MEAN',
        loss_params=None,
    ):
        super().__init__(
            model_name, objective_function, input_keys, loss_params=loss_params
        )
        if not events:
            raise ValueError('flood-event calibration requires at least one event')

        agg = str(event_loss_agg).upper()
        if agg not in EVENT_LOSS_AGGREGATORS:
            raise ValueError(
                f'unknown event_loss_agg {event_loss_agg!r}; '
                f'choose from: {sorted(EVENT_LOSS_AGGREGATORS)}'
            )
        self.event_loss_agg = agg
        self.aggregate_event_loss = EVENT_LOSS_AGGREGATORS[agg]

        self.warmup_length = warmup_length
        self.calib_events = []
        self.event_obs = []
        for ev in events:
            idx = np.asarray(ev['eval_idx'], dtype=int)
            if idx.size == 0:
                raise ValueError(f'event {ev["event_stem"]} has no flood_event steps')
            self.calib_events.append({'p_and_e': ev['p_and_e'], 'eval_idx': idx})
            self.event_obs.append(ev['qobs'][warmup_length + idx, 0, 0])

        self.true_obs = np.concatenate(self.event_obs).reshape(-1, 1, 1)

    def simulation(self, x):
        physical = self.denormalize_params(x)
        if physical is None:
            return self.invalid_simulation(self.true_obs)
        parts = []
        for ev in self.calib_events:
            qsim = self.simulator.simulate(
                ev['p_and_e'],
                physical,
                warmup_length=self.warmup_length,
            )
            parts.append(qsim[ev['eval_idx'], 0, 0])
        return np.concatenate(parts).reshape(-1, 1, 1)

    def evaluation(self):
        return self.true_obs

    def objectivefunction(self, simulation, evaluation, params=None):
        if np.isnan(simulation).all():
            return 1e12
        split_at = np.cumsum([len(obs) for obs in self.event_obs[:-1]])
        sim_parts = np.split(simulation[:, 0, 0], split_at)
        return self.aggregate_event_loss(self.event_obs, sim_parts, self.loss_fn)


# ---------------------------------------------------------------------------
# SCE-UA
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def sceua_progress_context():
    progress_re = re.compile(
        r'^\s*(\d+)\s+of\s+(\d+),\s+minimal objective function=([0-9.eE+-]+)'
    )
    out = sys.stdout
    buf = ''

    class StdoutFilter:
        def write(self, text):
            nonlocal buf
            if not text:
                return
            buf += text
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                if progress_re.search(line):
                    formatted = f'  {line.strip()}'
                    out.write(formatted + '\n')
                    append_basin_log(formatted)

        def flush(self):
            if buf and progress_re.search(buf):
                formatted = f'  {buf.strip()}'
                out.write(formatted + '\n')
                append_basin_log(formatted)
            buf = ''
            out.flush()

    sys.stdout = StdoutFilter()
    try:
        yield
    finally:
        sys.stdout = out


def near_init_population(center, nrows, npars, perturb):
    center = np.clip(np.asarray(center, dtype=np.float64).reshape(-1), 0.0, 1.0)
    x = np.zeros((nrows, npars))
    x[0, :] = center
    scale = max(float(perturb), 0.0)
    for i in range(1, nrows):
        if scale > 0.0:
            noise = np.random.uniform(-scale, scale, size=npars)
            x[i, :] = np.clip(center + noise, 0.0, 1.0)
        else:
            x[i, :] = center
    return x


class SCEUANearInit(spotpy.algorithms.sceua):
    def __init__(self, *args, init_norm, init_perturb=0.05, **kwargs):
        self.init_norm = np.asarray(init_norm, dtype=np.float64).reshape(-1)
        self.init_perturb = float(init_perturb)
        super().__init__(*args, **kwargs)

    def _sampleinputmatrix(self, nrows, npars):
        # SpotPy SCE-UA hook: seed population near donor params.
        return near_init_population(
            self.init_norm,
            nrows,
            npars,
            self.init_perturb,
        )


def extract_sceua_best(data, param_names):
    if data is None or len(data) == 0:
        raise RuntimeError('SCE-UA finished without saved runs in the database')
    best_idx = int(np.argmin(data['like1']))
    norm = np.array(
        [float(data[f'par{name}'][best_idx]) for name in param_names],
        dtype=np.float64,
    )
    return {
        'norm': norm,
        'objective': float(data['like1'][best_idx]),
        'n_evaluations': len(data),
        'best_evaluation': best_idx + 1,
    }


def run_sceua(setup, output_dir, algo_params, init_norm=None):
    rep = algo_params['rep']
    ngs = algo_params['ngs']
    kstop = algo_params['kstop']
    peps = algo_params['peps']
    pcento = algo_params['pcento']
    seed = algo_params['random_seed']
    log = get_logger('optimizers')
    log_section(
        log,
        f'▶ SCE-UA Calibration  [rep={rep}, ngs={ngs}, kstop={kstop}, '
        f'peps={peps}, pcento={pcento}]',
    )

    sampler_kw = {
        'dbname': output_dir,
        'dbformat': 'ram',
        'random_state': seed,
        'save_sim': False,
    }
    if init_norm is None:
        sampler_cls = spotpy.algorithms.sceua
    else:
        init_perturb = algo_params.get('init_perturb', 0.05)
        log.info('  Near-init warm start [perturb=%s].', init_perturb)
        sampler_cls = SCEUANearInit
        sampler_kw['init_norm'] = init_norm
        sampler_kw['init_perturb'] = init_perturb

    with sceua_progress_context():
        sampler = sampler_cls(setup, **sampler_kw)
        sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)
        best = extract_sceua_best(sampler.getdata(), setup.param_names)
    log.info('  Best objective %.3f after parameter search.', best['objective'])
    return best


ALGORITHM_RUNNERS = {'SCE-UA': run_sceua}
