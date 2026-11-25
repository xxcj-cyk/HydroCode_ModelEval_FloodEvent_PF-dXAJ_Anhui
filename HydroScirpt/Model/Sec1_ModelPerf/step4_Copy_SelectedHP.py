"""Sec1 step4 — copy Sec0 top1 HP runs (4 models × 9 seeds + ensemble) into Sec1."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SEC0 = ROOT / "HydroScirpt" / "Result" / "Sec0_ModelConfig"
SEC1 = ROOT / "HydroScirpt" / "Result" / "Sec1_ModelPerf"
SELECT_CSV = SEC1 / "Sec1_Selected_BestHP_Top5.csv"

SEEDS = ("1111", "2222", "3333", "4444", "5555", "6666", "7777", "8888", "9999")

MODEL_FOLDER = {
    "SeqRegLSTM": "SeqRegLSTM_Regional",
    "PFLoss-SeqRegLSTM": "PFLoss-SeqRegLSTM_Regional",
    "dXAJ-mz": "dXAJ-mz_Regional",
    "PFLoss-dXAJ-mz": "PFLoss-dXAJ-mz_Regional",
}


def load_recommended(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}; run step3_Select_BestHP.py first")
    df = pd.read_csv(csv_path)
    rec = df[df["recommended"].astype(str).str.lower().isin(["true", "1"])].copy()
    if rec.empty:
        raise ValueError(f"no recommended=True rows in {csv_path}")
    return rec.sort_values(["pair", "rank"]).groupby("pair", as_index=False).first()


def copy_dir(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dry_run:
        print(f"  DRY  {src.relative_to(SEC0)}  →  {dst.relative_to(SEC1)}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"  OK   {src.relative_to(SEC0)}  →  {dst.relative_to(SEC1)}")


def copy_model_hp(folder: str, hp_tag: str, dry_run: bool) -> int:
    n = 0
    for seed in SEEDS:
        name = f"{folder}_{hp_tag}_seed{seed}"
        copy_dir(SEC0 / folder / name, SEC1 / folder / name, dry_run=dry_run)
        n += 1
    ens_name = f"{folder}_{hp_tag}_Ensemble"
    copy_dir(
        SEC0 / f"{folder}_Ensemble" / ens_name,
        SEC1 / f"{folder}_Ensemble" / ens_name,
        dry_run=dry_run,
    )
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy Sec0 top1 HP runs into Sec1")
    ap.add_argument("--csv", type=str, default=str(SELECT_CSV))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rec = load_recommended(Path(args.csv))
    print(f"csv={args.csv}")
    print(f"Sec0 → Sec1: {SEC0} → {SEC1}")
    print(f"dry_run={args.dry_run}\n")

    total = 0
    for _, row in rec.iterrows():
        pair = str(row["pair"])
        hp_tag = str(row["hp_tag"])
        print("=" * 88)
        print(f"PAIR {pair}: hp_tag={hp_tag}")
        for model in [str(row["base_model"]), str(row["pf_model"])]:
            folder = MODEL_FOLDER[model]
            print(f"  model={model}  folder={folder}")
            total += copy_model_hp(folder, hp_tag, dry_run=args.dry_run)
        print()

    print("=" * 88)
    print(f"done  dirs={'planned' if args.dry_run else 'copied'}={total}  (expect 40 = 4 models × 10)")
    print("=" * 88)


if __name__ == "__main__":
    main()
