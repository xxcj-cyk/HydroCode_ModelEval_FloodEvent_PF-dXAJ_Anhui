# HydroScirpt

Paper experiment scripts for flood-event model evaluation on the Anhui dataset. Each script is a self-contained entry point that calls [HydroModels_PBM](../HydroModels_PBM) or [HydroModels_DLM](../HydroModels_DLM), writing results under `HydroScirpt/Result/`.

## Overview

Scripts are grouped by paper section under `Model/`:

| Section | Purpose |
|---------|---------|
| `Sec0_ModelConfig` | Regional HP sweep (4 models × 60 HP × 9 seeds), ensembles, overall summary |
| `Sec1_ModelPerf` | XAJ-mz local baselines; select & copy DLM showcase HP from Sec0 |
| `Sec2_LossSens` | PF-loss sensitivity: fixed Sec1 top1 HP × γ × μ × 9 seeds |

## Models

| Backend | Models | Typical use |
|---------|--------|-------------|
| PBM | `XAJ-mz_Anhui` | Local SCE-UA: RMSE and PF_PEAK80 (`wu168`) |
| DLM | `SeqRegLSTM`, `dXAJ-mz_Anhui` | Regional train, ensemble, PF-loss sensitivity |

Metrics include NSE, KGE, RMSE, HighRMSE, PFE, PTE (flood-event oriented).

## Data and paths

Default input:

```
/home/xxcj/Dataset/Processed_Dataset/CHINA/Anhui18_663
```

Update `data_cfgs.input_path` in each script for your environment. Flood-event windows are June–August 2024; `warmup_length` is typically 96–168 hours. Sec0 HP grids (`batch` / `warmup` / `lr`) are hardcoded in the scripts (no external CSV).

## Running

From the repository root:

```bash
# Sec0 — Regional HP sweep (seed 1111–3333→GPU2, 4444–6666→GPU1, 7777–9999→GPU0)
python HydroScirpt/Model/Sec0_ModelConfig/step1_SeqRegLSTM_Regional.py
python HydroScirpt/Model/Sec0_ModelConfig/step2_PFLoss-SeqRegLSTM_Regional.py
python HydroScirpt/Model/Sec0_ModelConfig/step3_dXAJ-mz_Regional.py
python HydroScirpt/Model/Sec0_ModelConfig/step4_PFLoss-dXAJ-mz_Regional.py

# Sec0 — 9-seed median ensembles, then overall tables
python HydroScirpt/Model/Sec0_ModelConfig/step5_SeqRegLSTM_Regional_Ensemble.py
python HydroScirpt/Model/Sec0_ModelConfig/step6_PFLoss-SeqRegLSTM_Regional_Ensemble.py
python HydroScirpt/Model/Sec0_ModelConfig/step7_dXAJ-mz_Regional_Ensemble.py
python HydroScirpt/Model/Sec0_ModelConfig/step8_PFLoss-dXAJ-mz_Regional_Ensemble.py
python HydroScirpt/Model/Sec0_ModelConfig/step9_Summarize_Overall.py

# Sec1 — XAJ-mz local + select/copy DLM showcase HP
python HydroScirpt/Model/Sec1_ModelPerf/step1_XAJ-mz_Local.py
python HydroScirpt/Model/Sec1_ModelPerf/step2_PFLoss-XAJ-mz_Local.py
python HydroScirpt/Model/Sec1_ModelPerf/step3_Select_BestHP.py
python HydroScirpt/Model/Sec1_ModelPerf/step4_Copy_SelectedHP.py

# Sec2 — PF-loss sensitivity on Sec1 top1 HP
python HydroScirpt/Model/Sec2_LossSens/step1_PFLoss-SeqRegLSTM_LossSens.py
python HydroScirpt/Model/Sec2_LossSens/step2_PFLoss-dXAJ_LossSens.py
```

Later steps depend on earlier outputs (Overall tables, selected HP CSV, Sec1 copies). Check path constants in each script before running downstream jobs.

## Directory layout

```
HydroScirpt/
  Model/
    Sec0_ModelConfig/   # step1–4 train, step5–8 ensemble, step9 summarize
    Sec1_ModelPerf/     # step1–2 XAJ local, step3 select HP, step4 copy
    Sec2_LossSens/      # step1 PF-LSTM, step2 PF-dXAJ
  Result/               # experiment outputs (gitignored)
```
