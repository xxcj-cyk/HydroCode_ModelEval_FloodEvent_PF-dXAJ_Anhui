"""Sec1 step3 — select LSTM / dXAJ showcase HP (Fig.4–10 rules).

Primary: MEDIAN × test × ensemble (PF − baseline)
  must   Δ|PFE| < 0 and ΔHF < 0 (HIGHRMSE80, mm)
  allow  ΔKGE ≥ -0.02
  fit    baseline |PFE|, HF and PF |PFE| ≤ cross-tag q75
  seed   same tag: n(Δ|PFE|<0)≥5/9 and n(ΔHF<0)≥5/9
  event  n(Δ|PFE|<0)/N ≥ 0.45; under-side median Δ|PFE| < 0
Sort: Δ|PFE| → ΔHF → seed N/9 → event% → under Δ|PFE| → pf|PFE| → ΔKGE

Reads Sec0 Overall tables; event diagnostics from ensemble event_metrics.csv.
Writes Result/Sec1_ModelPerf/Sec1_Selected_BestHP_Top5.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SEC0 = ROOT / "HydroScirpt" / "Result" / "Sec0_ModelConfig"
SEC1 = ROOT / "HydroScirpt" / "Result" / "Sec1_ModelPerf"
OUT_CSV = SEC1 / "Sec1_Selected_BestHP_Top5.csv"

STATS = ("MEDIAN", "MEAN")
SPLITS = ("train", "valid", "test")
DKGE_FLOOR = -0.02
SEED_MIN = 5
EVENT_IMPROVE_MIN = 0.45
HF_COL = "HIGHRMSE80"

PAIRS = {
    "lstm": {
        "title": "LSTM ↔ PF-LSTM",
        "base": "SeqRegLSTM",
        "pf": "PFLoss-SeqRegLSTM",
        "base_folder": "SeqRegLSTM_Regional",
        "pf_folder": "PFLoss-SeqRegLSTM_Regional",
        "current": "h128_dr01_wu096_b0512_lr001",
    },
    "dxaj": {
        "title": "dXAJ ↔ PF-dXAJ",
        "base": "dXAJ-mz",
        "pf": "PFLoss-dXAJ-mz",
        "base_folder": "dXAJ-mz_Regional",
        "pf_folder": "PFLoss-dXAJ-mz_Regional",
        "current": "h128_dr01_wu168_b0128_lr005",
    },
}

TOP_COLS = [
    "pair",
    "title",
    "rank",
    "recommended",
    "hp_tag",
    "base_model",
    "pf_model",
    "warmup",
    "batch_size",
    "learning_rate",
    "dPFE",
    "dHF",
    "dKGE",
    "base_KGE",
    "base_HF",
    "base_PFE",
    "pf_KGE",
    "pf_HF",
    "pf_PFE",
    "n_seed",
    "n_pfe_seed",
    "n_hf_seed",
    "n_both_seed",
    "n_event",
    "frac_pfe_improve",
    "under_side_dPFE_med",
]


def load_overall_tables() -> dict[tuple[str, str], pd.DataFrame]:
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    missing = []
    for stat in STATS:
        for split in SPLITS:
            path = SEC0 / f"Sec0_DLM_Overall_{stat}_{split}_Seed_and_Ensemble.csv"
            if path.is_file():
                tables[(stat, split)] = pd.read_csv(path)
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing Overall tables:\n  " + "\n  ".join(missing))
    return tables


def ensemble_by_hp(df: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = df[(df["run_type"] == "ensemble") & (df["model"] == model)]
    if sub.empty:
        raise ValueError(f"no ensemble rows for model={model}")
    return sub.set_index("hp_tag", drop=False)


def pair_deltas(df: pd.DataFrame, base: str, pf: str) -> pd.DataFrame:
    b = ensemble_by_hp(df, base)
    p = ensemble_by_hp(df, pf)
    common = sorted(set(b.index) & set(p.index))
    if not common:
        raise ValueError(f"no common hp_tag between {base} and {pf}")
    rows = []
    for tag in common:
        br, pr = b.loc[tag], p.loc[tag]
        rows.append(
            {
                "tag": tag,
                "warmup": int(br["warmup"]),
                "batch_size": int(br["batch_size"]),
                "learning_rate": float(br["learning_rate"]),
                "base_KGE": float(br["KGE"]),
                "base_HF": float(br[HF_COL]),
                "base_PFE": float(br["PFE_ABS"]),
                "pf_KGE": float(pr["KGE"]),
                "pf_HF": float(pr[HF_COL]),
                "pf_PFE": float(pr["PFE_ABS"]),
                "dKGE": float(pr["KGE"]) - float(br["KGE"]),
                "dHF": float(pr[HF_COL]) - float(br[HF_COL]),
                "dPFE": float(pr["PFE_ABS"]) - float(br["PFE_ABS"]),
            }
        )
    return pd.DataFrame(rows)


def seed_agreement(df: pd.DataFrame, base: str, pf: str, tag: str) -> dict:
    seeds = df[
        (df["run_type"] == "seed")
        & (df["hp_tag"] == tag)
        & (df["model"].isin([base, pf]))
    ].copy()
    seeds["seed_tag"] = seeds["seed_tag"].astype(str)
    b = seeds[seeds["model"] == base].set_index("seed_tag")
    p = seeds[seeds["model"] == pf].set_index("seed_tag")
    common = sorted(set(b.index) & set(p.index))
    if not common:
        return {
            "n_seed": 0,
            "n_pfe_seed": 0,
            "n_hf_seed": 0,
            "n_both_seed": 0,
            "med_dKGE_seed": np.nan,
            "med_dHF_seed": np.nan,
            "med_dPFE_seed": np.nan,
        }
    ddf = pd.DataFrame(
        {
            "dKGE": [float(p.loc[s, "KGE"]) - float(b.loc[s, "KGE"]) for s in common],
            "dHF": [float(p.loc[s, HF_COL]) - float(b.loc[s, HF_COL]) for s in common],
            "dPFE": [float(p.loc[s, "PFE_ABS"]) - float(b.loc[s, "PFE_ABS"]) for s in common],
        }
    )
    return {
        "n_seed": len(ddf),
        "n_pfe_seed": int((ddf["dPFE"] < 0).sum()),
        "n_hf_seed": int((ddf["dHF"] < 0).sum()),
        "n_both_seed": int(((ddf["dPFE"] < 0) & (ddf["dHF"] < 0)).sum()),
        "med_dKGE_seed": float(ddf["dKGE"].median()),
        "med_dHF_seed": float(ddf["dHF"].median()),
        "med_dPFE_seed": float(ddf["dPFE"].median()),
    }


def load_event_map(ens_dir: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not ens_dir.is_dir():
        return out
    for em in ens_dir.glob("Anhui_*/event_metrics.csv"):
        basin = em.parent.name
        df = pd.read_csv(em)
        if "period" in df.columns:
            df = df[df["period"].astype(str).str.lower() == "test"]
        for _, r in df.iterrows():
            out[(basin, str(r["event_id"]))] = {
                "PFE_ABS": float(r["PFE_ABS"]),
                "PFE": float(r["PFE"]),
            }
    return out


def event_diagnostics(pair: dict, tag: str) -> dict:
    base_dir = SEC0 / f"{pair['base_folder']}_Ensemble" / f"{pair['base_folder']}_{tag}_Ensemble"
    pf_dir = SEC0 / f"{pair['pf_folder']}_Ensemble" / f"{pair['pf_folder']}_{tag}_Ensemble"
    base_map = load_event_map(base_dir)
    pf_map = load_event_map(pf_dir)
    common = sorted(set(base_map) & set(pf_map))
    if not common:
        return {
            "n_event": 0,
            "frac_pfe_improve": np.nan,
            "under_side_dPFE_med": np.nan,
            "under_n": 0,
            "under_improve_frac": np.nan,
        }
    d_pfe = np.array([pf_map[k]["PFE_ABS"] - base_map[k]["PFE_ABS"] for k in common])
    under = [k for k in common if base_map[k]["PFE"] < 0]
    under_dp = np.array(
        [pf_map[k]["PFE_ABS"] - base_map[k]["PFE_ABS"] for k in under], dtype=float
    )
    return {
        "n_event": len(common),
        "frac_pfe_improve": float(np.mean(d_pfe < 0)),
        "under_side_dPFE_med": float(np.median(under_dp)) if len(under_dp) else np.nan,
        "under_n": len(under),
        "under_improve_frac": float(np.mean(under_dp < 0)) if len(under_dp) else np.nan,
    }


def enrich(pair: dict, ranked: pd.DataFrame, test_med: pd.DataFrame, use_event: bool) -> pd.DataFrame:
    rows = []
    for _, r in ranked.iterrows():
        row = {
            **r.to_dict(),
            **seed_agreement(test_med, pair["base"], pair["pf"], r["tag"]),
        }
        if use_event:
            row.update(event_diagnostics(pair, r["tag"]))
        else:
            row.update(
                {
                    "n_event": np.nan,
                    "frac_pfe_improve": np.nan,
                    "under_side_dPFE_med": np.nan,
                    "under_n": np.nan,
                    "under_improve_frac": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_pool(df: pd.DataFrame, use_event: bool) -> tuple[pd.DataFrame, dict]:
    pfe_q75 = float(np.quantile(df["base_PFE"], 0.75))
    hf_q75 = float(np.quantile(df["base_HF"], 0.75))
    m = df.copy()
    pass_must = (m["dPFE"] < 0) & (m["dHF"] < 0)
    pass_kge = m["dKGE"] >= DKGE_FLOOR
    pass_fit = (m["base_PFE"] <= pfe_q75) & (m["base_HF"] <= hf_q75) & (m["pf_PFE"] <= pfe_q75)
    pass_seed = (m["n_pfe_seed"] >= SEED_MIN) & (m["n_hf_seed"] >= SEED_MIN)
    pass_event = (
        (m["frac_pfe_improve"] >= EVENT_IMPROVE_MIN) & (m["under_side_dPFE_med"] < 0)
        if use_event
        else True
    )

    m["pass_all"] = pass_must & pass_kge & pass_fit & pass_seed & pass_event
    pool = m[m["pass_all"]].copy()
    if use_event:
        pool = pool.sort_values(
            by=[
                "dPFE",
                "dHF",
                "n_pfe_seed",
                "n_hf_seed",
                "frac_pfe_improve",
                "under_side_dPFE_med",
                "pf_PFE",
                "dKGE",
            ],
            ascending=[True, True, False, False, False, True, True, False],
        )
    else:
        pool = pool.sort_values(
            by=["dPFE", "dHF", "n_pfe_seed", "n_hf_seed", "pf_PFE", "dKGE"],
            ascending=[True, True, False, False, True, False],
        )
    meta = {
        "pfe_q75": pfe_q75,
        "hf_q75": hf_q75,
        "n_pass_must": int(pass_must.sum()),
        "n_pass_all": int(m["pass_all"].sum()),
    }
    return pool, meta


def fmt_row(r: pd.Series) -> str:
    bits = [
        str(r["tag"]),
        f"Δ|PFE|={r['dPFE']:+.2f}",
        f"ΔHF={r['dHF']:+.4f}",
        f"ΔKGE={r['dKGE']:+.4f}",
        f"pf|PFE|={r['pf_PFE']:.2f}",
        f"seed |PFE| {int(r['n_pfe_seed'])}/{int(r['n_seed'])} HF {int(r['n_hf_seed'])}/{int(r['n_seed'])}",
    ]
    if pd.notna(r.get("frac_pfe_improve", np.nan)):
        bits.append(f"event improve {r['frac_pfe_improve']*100:.0f}% (n={int(r['n_event'])})")
    if pd.notna(r.get("under_side_dPFE_med", np.nan)):
        bits.append(f"under Δ|PFE| med={r['under_side_dPFE_med']:+.2f}")
    return " | ".join(bits)


def print_crosscheck(tables: dict, base: str, pf: str, tag: str) -> None:
    print("    cross-check ensemble on all Overall tables:")
    for stat in STATS:
        for split in SPLITS:
            b = ensemble_by_hp(tables[(stat, split)], base)
            p = ensemble_by_hp(tables[(stat, split)], pf)
            if tag not in b.index or tag not in p.index:
                print(f"      {stat:6s} {split:5s}: missing tag")
                continue
            br, pr = b.loc[tag], p.loc[tag]
            print(
                f"      {stat:6s} {split:5s}: "
                f"KGE {float(br['KGE']):.4f}/{float(pr['KGE']):.4f} "
                f"({float(pr['KGE'])-float(br['KGE']):+.4f})  "
                f"HF {float(br[HF_COL]):.4f}/{float(pr[HF_COL]):.4f} "
                f"({float(pr[HF_COL])-float(br[HF_COL]):+.4f})  "
                f"|PFE| {float(br['PFE_ABS']):.2f}/{float(pr['PFE_ABS']):.2f} "
                f"({float(pr['PFE_ABS'])-float(br['PFE_ABS']):+.2f})"
            )


def pool_to_top_table(pair_key: str, pool: pd.DataFrame, top: int) -> pd.DataFrame:
    pair = PAIRS[pair_key]
    rows = []
    for i, (_, r) in enumerate(pool.head(top).iterrows(), 1):
        rows.append(
            {
                "pair": pair_key,
                "title": pair["title"],
                "rank": i,
                "recommended": i == 1,
                "hp_tag": r["tag"],
                "base_model": pair["base"],
                "pf_model": pair["pf"],
                "warmup": int(r["warmup"]),
                "batch_size": int(r["batch_size"]),
                "learning_rate": float(r["learning_rate"]),
                "dPFE": float(r["dPFE"]),
                "dHF": float(r["dHF"]),
                "dKGE": float(r["dKGE"]),
                "base_KGE": float(r["base_KGE"]),
                "base_HF": float(r["base_HF"]),
                "base_PFE": float(r["base_PFE"]),
                "pf_KGE": float(r["pf_KGE"]),
                "pf_HF": float(r["pf_HF"]),
                "pf_PFE": float(r["pf_PFE"]),
                "n_seed": int(r["n_seed"]),
                "n_pfe_seed": int(r["n_pfe_seed"]),
                "n_hf_seed": int(r["n_hf_seed"]),
                "n_both_seed": int(r["n_both_seed"]),
                "n_event": int(r["n_event"]) if pd.notna(r.get("n_event", np.nan)) else np.nan,
                "frac_pfe_improve": float(r["frac_pfe_improve"])
                if pd.notna(r.get("frac_pfe_improve", np.nan))
                else np.nan,
                "under_side_dPFE_med": float(r["under_side_dPFE_med"])
                if pd.notna(r.get("under_side_dPFE_med", np.nan))
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=TOP_COLS)


def run_pair(
    pair_key: str, tables: dict, top: int, use_event: bool
) -> tuple[str | None, pd.DataFrame]:
    pair = PAIRS[pair_key]
    test_med = tables[("MEDIAN", "test")]
    ranked = pair_deltas(test_med, pair["base"], pair["pf"])

    print("=" * 96)
    print(f"PAIR   : {pair['title']}")
    print("SOURCE : MEDIAN × test × ensemble")
    print(
        f"RULE   : Δ|PFE|<0 & ΔHF<0 ({HF_COL} mm) & ΔKGE≥-0.02 & fitness(q75) & "
        f"seed≥{SEED_MIN}/9"
        + (
            f" & event improve≥{EVENT_IMPROVE_MIN:.0%} & under-side med Δ|PFE|<0"
            if use_event
            else " (--no-event)"
        )
    )
    print("SORT   : Δ|PFE| → ΔHF → seed N/9 → event% → under Δ|PFE| → pf|PFE| → ΔKGE")
    print("Enriching diagnostics...")
    ranked = enrich(pair, ranked, test_med, use_event=use_event)
    pool, meta = select_pool(ranked, use_event=use_event)
    print(
        f"N tags={len(ranked)} | pass ΔPFE&ΔHF={meta['n_pass_must']} | "
        f"pass_all={meta['n_pass_all']} | fitness q75 |PFE|={meta['pfe_q75']:.2f} "
        f"HF={meta['hf_q75']:.4f}"
    )

    if pool.empty:
        print("NO candidate under full rule.")
        soft = ranked[(ranked["dPFE"] < 0) & (ranked["dHF"] < 0) & (ranked["dKGE"] >= DKGE_FLOOR)]
        soft = soft.sort_values(["dPFE", "dHF"])
        print("Near-miss (must Δ + KGE allow only):")
        for i, (_, r) in enumerate(soft.head(5).iterrows(), 1):
            print(f"  {i}. {fmt_row(r)}")
        return None, pd.DataFrame(columns=TOP_COLS)

    best = pool.iloc[0]
    print("\n>>> RECOMMENDED")
    print("   ", fmt_row(best))
    print_crosscheck(tables, pair["base"], pair["pf"], best["tag"])

    n_show = min(top, len(pool))
    print(f"\nTOP {n_show}:")
    for i, (_, r) in enumerate(pool.head(top).iterrows(), 1):
        mark = "  <-- RECOMMENDED" if i == 1 else ""
        print(f"  {i:2d}. {fmt_row(r)}{mark}")

    cur_tag = pair["current"]
    if cur_tag in set(ranked["tag"]):
        cur = ranked[ranked["tag"] == cur_tag].iloc[0]
        print("\nCURRENT_ADOPTED")
        print("   ", fmt_row(cur))
        if cur_tag in set(pool["tag"]):
            pos = int(np.where(pool["tag"].to_numpy() == cur_tag)[0][0] + 1)
            print(f"    in_pool=True  pool_rank={pos}")
        else:
            fails = []
            if not (cur["dPFE"] < 0):
                fails.append("Δ|PFE|<0")
            if not (cur["dHF"] < 0):
                fails.append("ΔHF<0")
            if not (cur["dKGE"] >= DKGE_FLOOR):
                fails.append("ΔKGE≥-0.02")
            if not (cur["n_pfe_seed"] >= SEED_MIN and cur["n_hf_seed"] >= SEED_MIN):
                fails.append("seed")
            if use_event:
                if not (cur.get("frac_pfe_improve", 0) >= EVENT_IMPROVE_MIN):
                    fails.append("event%")
                if not (cur.get("under_side_dPFE_med", 0) < 0):
                    fails.append("under")
            print(f"    in_pool=False  fails={fails or ['fitness/other']}")

    return str(best["tag"]), pool_to_top_table(pair_key, pool, top)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.4–10 HP selection for Sec1")
    ap.add_argument("--pair", choices=["both", "lstm", "dxaj"], default="both")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--no-event", action="store_true", help="skip event-level gates")
    args = ap.parse_args()

    tables = load_overall_tables()
    print(f"Loaded Overall tables from {SEC0}")
    print("Primary: MEDIAN / test / ensemble; other tables = cross-check only.\n")

    use_event = not args.no_event
    keys = ["lstm", "dxaj"] if args.pair == "both" else [args.pair]
    picks: dict[str, str | None] = {}
    top_frames: list[pd.DataFrame] = []
    for k in keys:
        best_tag, top_df = run_pair(k, tables, args.top, use_event=use_event)
        picks[k] = best_tag
        if not top_df.empty:
            top_frames.append(top_df)
        print()

    SEC1.mkdir(parents=True, exist_ok=True)
    out = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame(columns=TOP_COLS)
    out_path = OUT_CSV if args.top == 5 else SEC1 / f"Sec1_Selected_BestHP_Top{args.top}.csv"
    for c in [
        "learning_rate",
        "dPFE",
        "dHF",
        "dKGE",
        "base_KGE",
        "base_HF",
        "base_PFE",
        "pf_KGE",
        "pf_HF",
        "pf_PFE",
        "frac_pfe_improve",
        "under_side_dPFE_med",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 96)
    print("SUMMARY — suggested showcase HP (same tag for baseline & PF)")
    for k in keys:
        print(f"  {k:4s}  {PAIRS[k]['title']:22s}  →  {picks[k] or '(none)'}")
    print(f"wrote {out_path}  rows={len(out)}")
    print("=" * 96)


if __name__ == "__main__":
    main()
