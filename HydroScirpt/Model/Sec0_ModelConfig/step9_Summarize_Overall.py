"""Sec0 · step9：汇总 DLM overall MEDIAN / MEAN（9 seed + ensemble）。

读 Result/Sec0_ModelConfig 下各 seed / ensemble 的
best_metrics_{train,valid,test}.csv（basin_id 为空的 overall 行）。

输出 6 张表（Result/Sec0_ModelConfig/）：
  Sec0_DLM_Overall_{MEDIAN,MEAN}_{train,valid,test}_Seed_and_Ensemble.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SEC0 = ROOT / "HydroScirpt" / "Result" / "Sec0_ModelConfig"

SEEDS = ("1111", "2222", "3333", "4444", "5555", "6666", "7777", "8888", "9999")
SPLITS = ("train", "valid", "test")
STATS = ("MEDIAN", "MEAN")

MODELS = [
    ("SeqRegLSTM", "SeqRegLSTM_Regional"),
    ("PFLoss-SeqRegLSTM", "PFLoss-SeqRegLSTM_Regional"),
    ("dXAJ-mz", "dXAJ-mz_Regional"),
    ("PFLoss-dXAJ-mz", "PFLoss-dXAJ-mz_Regional"),
]

METRIC_COLS = [
    "NSE",
    "KGE",
    "RMSE",
    "HIGHRMSE",
    "HIGHRMSE70",
    "HIGHRMSE80",
    "HIGHRMSE90",
    "RMSE_Q",
    "HIGHRMSE_Q",
    "HIGHRMSE70_Q",
    "HIGHRMSE80_Q",
    "HIGHRMSE90_Q",
    "PFE",
    "PFE_ABS",
    "PTE",
    "PTE_ABS",
    "VE",
    "VE_ABS",
    "QR_PFE",
    "QR_VE",
    "QR_PTE",
    "QR",
]

SEED_RE = re.compile(
    r"^.+_(?P<hp_tag>h(?P<h>\d+)_dr(?P<dr>\d+)_wu(?P<wu>\d+)_b(?P<b>\d+)_lr(?P<lr>[0-9.]+))_seed(?P<seed>\d+)$"
)
ENS_RE = re.compile(
    r"^.+_(?P<hp_tag>h(?P<h>\d+)_dr(?P<dr>\d+)_wu(?P<wu>\d+)_b(?P<b>\d+)_lr(?P<lr>[0-9.]+))_Ensemble$"
)


def parse_hp(match: re.Match) -> dict:
    return {
        "hp_tag": match.group("hp_tag"),
        "warmup": int(match.group("wu")),
        "batch_size": int(match.group("b")),
        "learning_rate": float(f"0.{match.group('lr')}"),
    }


def overall_stat_row(path: Path, stat: str) -> pd.Series:
    df = pd.read_csv(path)
    empty = df["basin_id"].isna() | (df["basin_id"].astype(str).str.strip() == "")
    rows = df.loc[empty & (df["stat"].astype(str).str.upper() == stat.upper())]
    if rows.empty:
        raise ValueError(f"no overall {stat} row in {path}")
    return rows.iloc[0]


def metrics_row(run_dir: Path, split: str, stat: str, model: str, run_type: str, seed_tag: str, hp: dict) -> dict:
    overall = overall_stat_row(run_dir / f"best_metrics_{split}.csv", stat)
    row = {
        "model": model,
        "split": split,
        "stat": stat,
        "run_type": run_type,
        "seed_tag": seed_tag,
        **hp,
    }
    for col in METRIC_COLS:
        if col in overall.index:
            row[col] = float(overall[col])
    return row


def collect_model(model: str, folder: str, split: str, stat: str) -> pd.DataFrame:
    seed_root = SEC0 / folder
    ens_root = SEC0 / f"{folder}_Ensemble"
    rows: list[dict] = []

    for p in sorted(seed_root.iterdir()):
        if not p.is_dir():
            continue
        m = SEED_RE.match(p.name)
        if not m:
            continue
        seed = m.group("seed")
        if seed not in SEEDS:
            continue
        rows.append(metrics_row(p, split, stat, model, "seed", seed, parse_hp(m)))

    for p in sorted(ens_root.iterdir()):
        if not p.is_dir():
            continue
        m = ENS_RE.match(p.name)
        if not m:
            continue
        rows.append(metrics_row(p, split, stat, model, "ensemble", "Ensemble", parse_hp(m)))

    return pd.DataFrame(rows)


def validate_complete(df: pd.DataFrame) -> None:
    for model, _folder in MODELS:
        sub = df[df["model"] == model]
        tags = sorted(sub["hp_tag"].unique())
        if len(tags) != 60:
            raise ValueError(f"{model}: hp_tag={len(tags)} (expect 60)")
        for tag in tags:
            g = sub[sub["hp_tag"] == tag]
            seeds = set(g.loc[g["run_type"] == "seed", "seed_tag"].astype(str))
            n_ens = int((g["run_type"] == "ensemble").sum())
            if seeds != set(SEEDS) or n_ens != 1:
                raise ValueError(f"{model} {tag}: seeds={sorted(seeds)} n_ens={n_ens}")


def sort_table(df: pd.DataFrame) -> pd.DataFrame:
    model_order = {name: i for i, (name, _) in enumerate(MODELS)}
    seed_order = {s: i for i, s in enumerate(list(SEEDS) + ["Ensemble"])}
    out = df.copy()
    out["model_ord"] = out["model"].map(model_order)
    out["seed_ord"] = out["seed_tag"].map(seed_order)
    return (
        out.sort_values(["model_ord", "warmup", "batch_size", "learning_rate", "seed_ord"])
        .drop(columns=["model_ord", "seed_ord"])
        .reset_index(drop=True)
    )


def main() -> None:
    SEC0.mkdir(parents=True, exist_ok=True)
    for stat in STATS:
        for split in SPLITS:
            frames = [collect_model(model, folder, split, stat) for model, folder in MODELS]
            df = sort_table(pd.concat(frames, ignore_index=True))
            validate_complete(df)
            out = SEC0 / f"Sec0_DLM_Overall_{stat}_{split}_Seed_and_Ensemble.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(
                f"wrote {out.name} rows={len(df)} "
                f"seed={(df.run_type == 'seed').sum()} "
                f"ens={(df.run_type == 'ensemble').sum()}"
            )


if __name__ == "__main__":
    main()
