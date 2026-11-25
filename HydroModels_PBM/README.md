# HydroModels_PBM

Process-based hydrological modeling workflow for basin-scale calibration, evaluation, and parameter transfer. This package mirrors the layout of [HydroModels_DLM](../HydroModels_DLM) but uses classical PBM solvers and global optimization instead of deep learning.

## Overview

HydroModels_PBM provides a unified pipeline for:

- Loading daily or flood-event CSV time series with configurable variable mapping
- Calibrating PBM parameters with SCE-UA
- Simulating streamflow with warmup and explicit calibration / validation periods
- Evaluating NSE, KGE, RMSE, CORR, R2, MBE, PFE, PTE, VE, FHV, FLV, and related metrics
- Transferring calibrated parameters across basins (pair-list experiments)

All calibrated parameter names in `model_config.py` use **uppercase** keys (e.g. `X1`, `CN1`, `THETA`).

## Supported models (25)

| Family | Base | CemaNeige | mz | mz-CemaNeige |
|--------|------|-----------|-----|--------------|
| API | `API` | `API-CemaNeige` | — | — |
| GR4J | `GR4J` | `GR4J-CemaNeige` | `GR4J-mz` | `GR4J-mz-CemaNeige` |
| GR5J | `GR5J` | `GR5J-CemaNeige` | `GR5J-mz` | `GR5J-mz-CemaNeige` |
| GR6J | `GR6J` | `GR6J-CemaNeige` | `GR6J-mz` | `GR6J-mz-CemaNeige` |
| HBV | `HBV` | `HBV-CemaNeige` | `HBV-mz` | `HBV-mz-CemaNeige` |
| HYMOD | `HYMOD` | `HYMOD-CemaNeige` | — | — |
| TANK | `TANK` | `TANK-CemaNeige` | — | — |
| XAJ | `XAJ` | — | `XAJ-mz` | `XAJ-mz-CemaNeige` |

**Variants:**

- **CemaNeige** — snow module; drivers require `P`, `PET`, `T`. Snow parameters: `CN1`, `CN2`.
- **mz** — multi-zone routing with a Gamma unit hydrograph (`A`, `THETA`; length fixed at 48 steps, not calibrated). `HBV-mz` replaces triangle routing (`MAXBAS`) with this unit hydrograph.
- Shared implementations live in `hydromodels_pbm/model/cemaneige.py` and `hydromodels_pbm/model/mz.py`.

Full registry: `hydromodels_pbm/model/model_dict.py` (`MODEL_DICT`).

## Data format

Experiments expect preprocessed basin CSVs under `input_path`, with columns mapped through `data_cfgs.variables`. Periods are configured explicitly:

- `calib_period` — optimization window
- `valid_period` — hold-out evaluation window
- `warmup_length` — spin-up steps before each simulation window

CemaNeige models must include temperature in `dynamic_inputs`:

```python
"dynamic_inputs": {
    "P": "precipitation_column",
    "PET": "pet_column",
    "T": "temperature_column",  # required for *-CemaNeige
}
```

Both long-term daily data and flood-event subsets are supported via the data loader.

## Workflow

Each experiment script defines a config dict and calls `run_script_experiment()` from `hydromodels_pbm.workflow.pipeline`.

Typical outputs under `output_dir`:

- `best_params_all.json` — calibrated parameters per basin (uppercase keys)
- Per-basin `run_config.json`, logs, metrics CSVs, and simulated time series
- Experiment-level metric summaries across calibration / validation periods

For parameter transfer across basins, use `run_pair_list_experiment()` with either:

- `evaluation_cfgs.params_json` — fixed donor parameters (transplant), or
- `evaluation_cfgs.init_params_json` — warm-start recalibration per target basin

and a `transplant_list_csv` listing `target_basin_id,source_basin_id` pairs.

## Experiment scripts

Ready-to-run scripts for this project live under `../HydroScirpt/Model/Sec1_ModelPerf/`:

| Script | Objective | Output |
|--------|-----------|--------|
| `step1_XAJ-mz_Local.py` | RMSE | `XAJ-mz_Local/XAJ-mz_Local_wu168` |
| `step2_PFLoss-XAJ-mz_Local.py` | PF_PEAK80 | `PFLoss-XAJ-mz_Local/PFLoss-XAJ-mz_Local_wu168` |

```bash
cd HydroScript_ModelEval_FloodEvent_PF-dXAJ_Anhui
python HydroScirpt/Model/Sec1_ModelPerf/step1_XAJ-mz_Local.py
python HydroScirpt/Model/Sec1_ModelPerf/step2_PFLoss-XAJ-mz_Local.py
```

Edit `input_path`, `output_dir`, `basin_ids`, and period settings in each script before running.

## Package layout

```
hydromodels_pbm/
  config/       model_config (bounds, input keys), run_config
  dataset/      CSV loading, preprocessing, flood-event support
  model/        model families (API/, GR4J/, …), cemaneige.py, mz.py, model_dict.py
  workflow/     calibrate, simulate, evaluate, pipeline
  utils/        logging, normalization, runtime helpers
```

## Dependencies

NumPy, Pandas, SpotPy (SCE-UA), and Numba (JIT warmup). No PyTorch required.
