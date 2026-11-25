# HydroModels_DLM

Deep-learning hydrological modeling workflow aligned with [HydroModels_PBM](../HydroModels_PBM). Supports pure data-driven LSTM models and physics-guided models where an LSTM predicts bounded PBM parameters.

## Overview

HydroModels_DLM covers the full train–evaluate–transfer loop:

- **Data-driven**: `SeqRegLSTM` — LSTM maps dynamic forcings (+ optional static attributes) to streamflow
- **Physics-guided**: differentiable PBM cores (`dGR4J`, `dXAJ`, `dXAJ-mz`, …) — LSTM outputs parameters; the PBM stays in the autograd graph
- **Training strategies**: `RegionalTrain` (one shared model across basins) or `LocalTrain` (one model per basin)
- **Transfer learning**: zero-shot, fine-tune, and scaler reuse via `workflow/transfer.py`
- **Training losses**: `RMSE` and peak-focused `PF_PEAK80` (aligned with PBM). Evaluation metrics (NSE, KGE, HighRMSE, PFE, …) are separate from the training objective.

Data uses the same CSV layout as PBM, with explicit `train_period`, `valid_period`, and `test_period`. Flood-event scripts typically use short warmup (e.g. 96–168 h) and `event_loss_agg=LENGTH`.

## Supported models (7)

| Type | Models |
|------|--------|
| Data-driven | `SeqRegLSTM` |
| Physics-guided | `dGR4J`, `dGR4J-CemaNeige`, `dXAJ`, `dXAJ-mz`, `dXAJ-mz_Anhui`, `dXAJ-mz-CemaNeige` |

Full registry: `hydromodels_dlm/model/model_dict.py` (`MODEL_DICT`). Naming mirrors PBM with a `d` prefix. This paper repo’s HydroScirpt experiments use `SeqRegLSTM` and `dXAJ-mz_Anhui`.

## Physics-guided architecture

Training follows a two-stage differentiable design:

1. LSTM reads the full warmup + forecast window (`input_layout: dynamic_static`) and maps the last hidden state to bounded PBM parameters (`param_mode` default: `window`).
2. The PBM forward pass produces simulated discharge; streamflow error backpropagates into LSTM weights.

Performance options:

- Basin-wise vectorization (unit hydro weights, `conv1d` routing)
- Truncated backprop via `training_cfgs.pbm_grad_steps` (Anhui paper: `G=24`; `0` = full sequence)
- Numba CPU warmup for the non-differentiable prefix with PyTorch on the differentiable tail

## Environment

Use a conda env with a **CUDA build of PyTorch** (not the default CPU wheel):

```bash
conda activate hydro
cd HydroModels_DLM
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If CUDA torch is missing:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

On Windows, `configure_runtime()` in `hydromodels_dlm` handles OpenMP / MKL conflicts automatically.

## Outputs

**RegionalTrain** — shared artifacts under `output_dir/<experiment_name>/`:

- `best_model.pth`, `normalization_scaler.json`, `training_log.txt`
- Per-basin subdirs with `best_metrics.csv`, `timeseries.csv`, etc.
- Experiment-level `best_metrics_{train,valid,test}.csv` with a `MEDIAN` row

**LocalTrain** — per-basin dirs under `output_dir/<experiment_name>/<basin_id>/` with their own checkpoint and scaler; experiment root holds aggregated metric CSVs.

## Experiment scripts

Ready-to-run scripts for this project live under `../HydroScirpt/Model/`:

| Section | Scripts | Models / notes |
|---------|---------|----------------|
| `Sec0_ModelConfig` | `step1`–`step4_*Regional.py` | HP sweep: SeqRegLSTM / PFLoss-SeqRegLSTM / dXAJ-mz / PFLoss-dXAJ-mz |
| `Sec1_ModelPerf` | `train_sec1_*9seed.py`, `*_ensemble.py` | Paper ensembles (top-1 HP × 9 seeds) |
| `Sec2_LossSens` | `train_PFLoss-*_LossSens.py` | PF γ×μ sensitivity |

```bash
cd HydroScript_ModelEval_FloodEvent_PF-dXAJ_Anhui
python HydroScirpt/Model/Sec0_ModelConfig/step1_SeqRegLSTM_Regional.py
python HydroScirpt/Model/Sec0_ModelConfig/step4_PFLoss-dXAJ-mz_Regional.py
```

Edit `input_path`, HP lists, and seed→GPU maps in each script before running. Set `model_hyperparam.input_size` / `output_size` to match data dimensions (`output_size` for dXAJ via `PBM_PARAM_BOUNDS`).

## Package layout

```
hydromodels_dlm/
  config/       run_config, model_config (PBM parameter bounds)
  dataset/      loading, normalization, sliding-window Dataset
  model/        SeqRegLSTM, dGR4J, dXAJ(+mz), physics_guided wrappers
  workflow/     train, evaluate, transfer, pipeline, losses, metrics
  utils/        logging, GPU, random seed
```

Entry points: `run_script_experiment()` and `run_script_main()` in `workflow/pipeline.py`.

## Dependencies

PyTorch (CUDA), NumPy, Pandas, and Numba (optional JIT warmup). See `requirements.txt`.
