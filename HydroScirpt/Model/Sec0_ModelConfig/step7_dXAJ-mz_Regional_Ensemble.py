"""Sec0 step7 — dXAJ-mz_Regional | 9-seed median ensemble × 60 HP."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HydroModels_DLM"))

from hydromodels_dlm.utils.logging import setup_logging
from hydromodels_dlm.workflow.artifacts import (
    EVENT_METRICS,
    PERIOD_METRICS_FILES,
    PERIOD_ORDER,
    STAT_ORDER,
    save_flood_period_csv,
    summarize_flood_regional,
)
from hydromodels_dlm.workflow.evaluate import (
    basin_area_km2_from_member_metrics,
    compute_metrics,
)
from hydromodels_dlm.workflow.metrics import FLOOD_EVENT_METRICS

PREFIX = "dXAJ-mz_Regional"
SINGLE_ROOT = ROOT / "HydroScirpt/Result/Sec0_ModelConfig" / PREFIX
ENSEMBLE_ROOT = ROOT / "HydroScirpt/Result/Sec0_ModelConfig" / f"{PREFIX}_Ensemble"
SEEDS = (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999)
METRICS = list(FLOOD_EVENT_METRICS)
HP_RE = re.compile(
    rf"^{re.escape(PREFIX)}_(?P<hp>h\d+_dr\d+_wu\d+_b\d+_lr[0-9.]+)_seed(?P<seed>\d+)$"
)


def list_basin_dirs(exp_dir: Path) -> list[Path]:
    return sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("Anhui_"))


def discover_hp_groups() -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    if not SINGLE_ROOT.is_dir():
        return {}
    for path in sorted(SINGLE_ROOT.iterdir()):
        if not path.is_dir():
            continue
        m = HP_RE.match(path.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed not in SEEDS:
            continue
        if not (path / "best_metrics_test.csv").is_file():
            continue
        groups[m.group("hp")][seed] = path
    return dict(groups)


def load_aligned_qsim(member_dirs: list[Path], basin_name: str):
    ref = pd.read_csv(member_dirs[0] / basin_name / "timeseries.csv")
    ref_obs = ref[["time", "period", "event_id", "Qobs"]]
    cols = [pd.to_numeric(ref["Qsim"], errors="coerce").to_numpy()]
    for exp_dir in member_dirs[1:]:
        ts = pd.read_csv(exp_dir / basin_name / "timeseries.csv")
        if len(ts) != len(ref) or not ref_obs.equals(ts[ref_obs.columns]):
            raise ValueError(f"Timeseries mismatch: {exp_dir.name}/{basin_name}")
        cols.append(pd.to_numeric(ts["Qsim"], errors="coerce").to_numpy())
    return ref, np.column_stack(cols)


def format_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in METRICS:
        if name not in out.columns:
            continue
        prec = 2 if name == "PTE" else 4
        out[name] = out[name].map(lambda value: f"{float(value):.{prec}f}")
    return out


def write_basin_metrics(basin_dir, basin_id, timeseries, area_km2=None):
    event_rows = []
    period_stats: dict = {}
    for period in PERIOD_ORDER:
        period_df = timeseries.loc[timeseries["period"].astype(str).str.lower() == period]
        if period_df.empty:
            continue
        event_metric_rows = []
        for event_id, group in period_df.groupby("event_id", sort=False):
            metrics = compute_metrics(
                group["Qsim"].to_numpy(dtype=float),
                group["Qobs"].to_numpy(dtype=float),
                METRICS,
                area_km2=area_km2,
            )
            event_metric_rows.append(metrics)
            event_rows.append(
                {"period": period, "basin_id": basin_id, "event_id": event_id, **metrics}
            )
        period_stats[period] = {}
        for stat in STAT_ORDER:
            reducer = np.nanmedian if stat == "MEDIAN" else np.nanmean
            period_stats[period][stat] = {
                name: float(reducer([row[name] for row in event_metric_rows]))
                for name in METRICS
            }

    format_metric_frame(pd.DataFrame(event_rows)).to_csv(
        basin_dir / EVENT_METRICS, index=False, encoding="utf-8-sig"
    )
    basin_rows = []
    for stat in STAT_ORDER:
        for period in PERIOD_ORDER:
            metrics = period_stats.get(period, {}).get(stat)
            if metrics:
                basin_rows.append(
                    {"stat": stat, "period": period, "basin_id": basin_id, **metrics}
                )
    format_metric_frame(pd.DataFrame(basin_rows)).to_csv(
        basin_dir / "best_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return basin_rows


def write_ensemble_run(member_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    basin_names = [p.name for p in list_basin_dirs(member_dirs[0])]
    rows_by_period = {period: [] for period in PERIOD_ORDER}

    for basin_name in basin_names:
        basin_dir = out_dir / basin_name
        basin_dir.mkdir(parents=True, exist_ok=True)
        ref, qmat = load_aligned_qsim(member_dirs, basin_name)
        ts = ref.copy()
        ts["Qsim"] = np.median(qmat, axis=1)
        ts.to_csv(basin_dir / "timeseries.csv", index=False, encoding="utf-8-sig")
        area = basin_area_km2_from_member_metrics(member_dirs, basin_name)
        for row in write_basin_metrics(basin_dir, basin_name, ts, area):
            rows_by_period[row["period"]].append(row)

    summary_rows = []
    for period in PERIOD_ORDER:
        basin_rows = rows_by_period[period]
        if not basin_rows:
            continue
        period_rows = [{k: v for k, v in row.items() if k != "period"} for row in basin_rows]
        regional_rows = summarize_flood_regional(out_dir, basin_names, period)
        save_flood_period_csv(
            out_dir / PERIOD_METRICS_FILES[period], period_rows, regional_rows
        )
        df = pd.read_csv(out_dir / PERIOD_METRICS_FILES[period])
        stat_col = "stat" if "stat" in df.columns else df.columns[0]
        empty = df["basin_id"].isna() | (df["basin_id"].astype(str).str.strip() == "")
        for _, r in df.loc[empty].iterrows():
            summary_rows.append(
                {
                    "period": period,
                    "stat": str(r[stat_col]).upper(),
                    **{name: r[name] for name in METRICS if name in r.index},
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "ensemble_summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {"member": [p.name for p in member_dirs], "seed": list(SEEDS)}
    ).to_csv(out_dir / "ensemble_members.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    setup_logging()
    ENSEMBLE_ROOT.mkdir(parents=True, exist_ok=True)
    groups = discover_hp_groups()
    complete = {hp: seeds for hp, seeds in groups.items() if set(seeds) >= set(SEEDS)}
    print("=" * 60)
    print(f"Sec0 step7 — {PREFIX} | 9-seed median ensemble")
    print(f"single={SINGLE_ROOT}")
    print(f"ensemble={ENSEMBLE_ROOT}")
    print(f"HP found={len(groups)}  complete={len(complete)}")
    print("-" * 60)

    done = 0
    skipped = 0
    for hp in sorted(complete):
        out_dir = ENSEMBLE_ROOT / f"{PREFIX}_{hp}_Ensemble"
        if (out_dir / "ensemble_summary_metrics.csv").is_file():
            skipped += 1
            continue
        member_dirs = [complete[hp][s] for s in SEEDS]
        print(f"  -> {out_dir.name}", flush=True)
        write_ensemble_run(member_dirs, out_dir)
        done += 1

    print("=" * 60)
    print(f"Finished: wrote={done} skipped={skipped} complete_hp={len(complete)}")


if __name__ == "__main__":
    main()
