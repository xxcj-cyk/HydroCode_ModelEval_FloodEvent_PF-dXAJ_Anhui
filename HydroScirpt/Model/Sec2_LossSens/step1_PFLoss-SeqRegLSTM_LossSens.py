"""Sec2 step1 — PFLoss-SeqRegLSTM LossSens | Sec1 top1 HP × γ×μ × 9 seeds."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HydroModels_DLM"))

from hydromodels_dlm.config.run_config import merge_config, output_dir
from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.metrics import FLOOD_EVENT_METRICS
from hydromodels_dlm.workflow.pipeline import run_script_experiment

# --- knobs ---
WORKERS_PER_GPU = 10
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
PEAK_RATIO_LIST = [0.7, 0.8, 0.9]
WIN_WEIGHT_LIST = [0.05, 0.1, 0.2, 0.5, 1.0]

# Sec1 top1 HP (fixed)
WU, BS, LR = 120, 128, 0.001
HP_TAG = "h128_dr01_wu120_b0128_lr001"

PREFIX = "PFLoss-SeqRegLSTM_Regional_LossSens"
SEC1_PREFIX = "PFLoss-SeqRegLSTM_Regional"
OUT_REL = f"HydroScirpt/Result/Sec2_LossSens/{PREFIX}"
SEC1_DIR = ROOT / "HydroScirpt/Result/Sec1_ModelPerf" / SEC1_PREFIX


def pr_tag(peak_ratio: float) -> str:
    return f"pr{int(round(peak_ratio * 10)):02d}"


def ww_tag(win_weight: float) -> str:
    return f"ww{int(round(win_weight * 100)):02d}"


def run_name(pr: float, ww: float, seed: int) -> str:
    return f"{HP_TAG}_seed{seed}_{pr_tag(pr)}_{ww_tag(ww)}"


def is_baseline(pr: float, ww: float) -> bool:
    return abs(pr - 0.8) < 1e-9 and abs(ww - 0.1) < 1e-9


def result_done(pr: float, ww: float, seed: int) -> bool:
    name = f"{PREFIX}_{run_name(pr, ww, seed)}"
    return (ROOT / OUT_REL / name / "best_metrics_valid.csv").is_file()


def try_copy_baseline(seed: int) -> bool:
    src = SEC1_DIR / f"{SEC1_PREFIX}_{HP_TAG}_seed{seed}"
    dst = ROOT / OUT_REL / f"{PREFIX}_{run_name(0.8, 0.1, seed)}"
    if (dst / "best_metrics_valid.csv").is_file():
        return True
    if not (src / "best_metrics_valid.csv").is_file():
        print(f"  WARN missing Sec1 baseline: {src.name}", flush=True)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied Sec1 → {dst.name}", flush=True)
    return True


def build_config(pr: float, ww: float, seed: int) -> dict:
    name = run_name(pr, ww, seed)
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
            "warmup_length": WU,
            "train_period": ["2024-06-01 00:00:00", "2024-06-30 23:00:00"],
            "valid_period": ["2024-07-01 00:00:00", "2024-07-31 23:00:00"],
            "test_period": ["2024-08-01 00:00:00", "2024-08-31 23:00:00"],
        },
        "model_cfgs": {
            "model_name": "SeqRegLSTM",
            "model_hyperparam": {
                "input_size": 22,
                "output_size": 1,
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
                "overall_weight": 1.0,
                "win_weight": ww,
                "peak_ratio": pr,
                "peak_soft": False,
                "peak_excess": True,
                "peak_underonly": True,
                "peak_rel_excess": False,
            },
            "event_loss_agg": "LENGTH",
            "event_batch_size": 8,
            "optimizer_name": "Adam",
            "learning_rate": LR,
            "learning_rate_decay": 0.98,
            "batch_size": BS,
            "epochs": 50,
            "patience": 6,
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
    pr, ww, seed = job
    name = run_name(pr, ww, seed)
    if result_done(pr, ww, seed):
        return f"skip:{name}"
    if is_baseline(pr, ww) and try_copy_baseline(seed):
        return f"copy:{name}"
    try:
        overrides = build_config(pr, ww, seed)
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

    for seed in RS_LIST:
        try_copy_baseline(seed)

    sweep = list(product(PEAK_RATIO_LIST, WIN_WEIGHT_LIST, RS_LIST))
    todo = [j for j in sweep if not result_done(*j)]
    print("=" * 60)
    print(f"Sec2 step1 — {PREFIX} | PF_PEAK80 LossSens")
    print(f"HP={HP_TAG}")
    print(f"γ={PEAK_RATIO_LIST}  μ={WIN_WEIGHT_LIST}")
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

    counts = {"ok": 0, "copy": 0, "skip": 0, "fail": 0}
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
