"""Sec0 step4 — PFLoss-dXAJ-mz Regional | PF_PEAK80 | G=24 | 60 HP × 9 seeds."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HydroModels_DLM"))

from hydromodels_dlm.config.model_config import PBM_PARAM_BOUNDS
from hydromodels_dlm.config.run_config import merge_config, output_dir
from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.metrics import FLOOD_EVENT_METRICS
from hydromodels_dlm.workflow.pipeline import run_script_experiment

# PF_PEAK80
OVERALL_W = 1.0
WIN_W = 0.1
PEAK_RATIO = 0.8
PEAK_EXCESS = True
PEAK_UNDERONLY = True

# --- knobs ---
WORKERS_PER_GPU = 4
RS_LIST = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
SEED_DEVICE = {
    1111: 2,
    2222: 2,
    3333: 2,
    4444: 1,
    5555: 1,
    6666: 1,
    7777: 0,
    8888: 0,
    9999: 0,
}
BATCH_SIZE_LIST = [32, 64, 128, 256, 512]
WARMUP_LENGTH_LIST = [96, 120, 144, 168]
LR_LIST = [0.005, 0.001, 0.0005]

PREFIX = "PFLoss-dXAJ-mz_Regional"
OUT_REL = f"HydroScirpt/Result/Sec0_ModelConfig/{PREFIX}"
MODEL_NAME = "dXAJ-mz_Anhui"


def run_name(warmup: int, batch: int, lr: float, seed: int) -> str:
    tag = str(lr)
    if tag.startswith("0."):
        tag = tag[2:].rstrip("0") or "0"
    return f"h128_dr01_wu{warmup:03d}_b{batch:04d}_lr{tag}_seed{seed}"


def result_done(warmup: int, batch: int, lr: float, seed: int) -> bool:
    name = f"{PREFIX}_{run_name(warmup, batch, lr, seed)}"
    return (ROOT / OUT_REL / name / "best_metrics_test.csv").is_file()


def build_config(warmup: int, batch: int, lr: float, seed: int) -> dict:
    name = run_name(warmup, batch, lr, seed)
    return {
        "data_cfgs": {
            "input_path": r"/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663",
            "output_dir": OUT_REL,
            "basin_ids": None,
            "variables": {
                "dynamic_inputs": {"P": "p_anhui", "PET": "pet_anhui"},
                "dynamic_outputs": {"Q": "streamflow_obs_mm"},
                "static_attributes": [
                    "Pmean",
                    "Area",
                    "swc_pc_syr",
                    "sgr_dk_sav",
                    "ele_mt_sav",
                    "slp_dg_sav",
                    "pet_mm_syr",
                    "aet_mm_syr",
                    "tmp_dc_syr",
                    "cmi_ix_syr",
                    "snw_pc_syr",
                    "ari_ix_sav",
                    "for_pc_sse",
                    "crp_pc_sse",
                    "lit_cl_smj",
                    "soc_th_sav",
                    "kar_pc_sse",
                    "snd_pc_sav",
                    "slt_pc_sav",
                    "cly_pc_sav",
                ],
            },
            "scaler_params": {
                "log1p_zscore": ["P", "PET"],
                "prcp_log1p_zscore": ["Q"],
            },
            "forecast_length": 1,
            "input_layout": "dynamic_static",
            "warmup_length": warmup,
            "train_period": ["2024-06-01 00:00:00", "2024-06-30 23:00:00"],
            "valid_period": ["2024-07-01 00:00:00", "2024-07-31 23:00:00"],
            "test_period": ["2024-08-01 00:00:00", "2024-08-31 23:00:00"],
        },
        "model_cfgs": {
            "model_name": MODEL_NAME,
            "model_hyperparam": {
                "input_size": 22,
                "output_size": len(PBM_PARAM_BOUNDS[MODEL_NAME]),
                "hidden_size": 128,
                "dropout": 0.1,
                "input_proj": None,
            },
        },
        "training_cfgs": {
            "strategy": "RegionalTrain",
            "experiment_name": f"{PREFIX}_{name}",
            "loss_function": "PF_PEAK80",
            "loss_params": {
                "overall_weight": OVERALL_W,
                "win_weight": WIN_W,
                "peak_ratio": PEAK_RATIO,
                "peak_soft": False,
                "peak_excess": PEAK_EXCESS,
                "peak_underonly": PEAK_UNDERONLY,
                "peak_rel_excess": False,
            },
            "event_loss_agg": "LENGTH",
            "event_batch_size": 4,
            "optimizer_name": "Adam",
            "learning_rate": lr,
            "learning_rate_decay": 0.98,
            "batch_size": batch,
            "epochs": 50,
            "patience": 6,
            "pbm_grad_steps": 24,
            "random_seed": seed,
            "device": SEED_DEVICE[seed],
            "num_workers": 0,
        },
        "evaluation_cfgs": {
            "metrics": list(FLOOD_EVENT_METRICS),
            "eval_mode": "sliding",
        },
        "run_cfgs": {"skip_train": False},
    }


def run_single(job: tuple) -> str:
    warmup, batch, lr, seed = job
    name = run_name(warmup, batch, lr, seed)
    if result_done(warmup, batch, lr, seed):
        return f"skip:{name}"
    try:
        overrides = build_config(warmup, batch, lr, seed)
        print(f"  -> cuda:{SEED_DEVICE[seed]}  {output_dir(merge_config(overrides))}", flush=True)
        run_script_experiment(overrides)
        return f"ok:{name}"
    except Exception as exc:
        traceback.print_exc()
        return f"fail:{name}:{exc}"


def main() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    setup_logging()
    (ROOT / OUT_REL).mkdir(parents=True, exist_ok=True)

    sweep = list(product(WARMUP_LENGTH_LIST, BATCH_SIZE_LIST, LR_LIST, RS_LIST))
    todo = [j for j in sweep if not result_done(*j)]
    print("=" * 60)
    print(f"Sec0 step4 — {PREFIX} | PF_PEAK80 | G=24")
    print(f"seed→device={SEED_DEVICE}")
    print(f"workers/GPU={WORKERS_PER_GPU}  total={len(sweep)}  remaining={len(todo)}")
    print(f"out={OUT_REL}")
    print("-" * 60)
    if not todo:
        print("nothing to do")
        return

    by_device: dict[int, list[tuple]] = {}
    for job in todo:
        by_device.setdefault(SEED_DEVICE[job[-1]], []).append(job)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    ctx = mp.get_context("spawn")
    executors = []
    futures = {}
    try:
        for device, jobs in by_device.items():
            n_workers = min(WORKERS_PER_GPU, len(jobs))
            ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
            executors.append(ex)
            print(f"GPU {device}: {len(jobs)} jobs, workers={n_workers}", flush=True)
            for job in jobs:
                futures[ex.submit(run_single, job)] = job
        for fut in as_completed(futures):
            result = fut.result()
            tag = result.split(":", 1)[0]
            counts[tag] = counts.get(tag, 0) + 1
            print(f"[Progress] {counts} / {len(todo)}  last={result}", flush=True)
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    print("=" * 60)
    print(f"Finished: {counts} / {len(todo)}")


if __name__ == "__main__":
    main()
