# HydroCode_ModelEval_FloodEvent_PF-dXAJ_Anhui

Public code archive for flood-event hydrological model evaluation on Anhui basins, comparing process-based models (PBM), deep-learning models (DLM), and peak-focused training losses (PF-LSTM / PF-dXAJ).

This repository vendors [HydroModels_PBM](HydroModels_PBM/) and [HydroModels_DLM](HydroModels_DLM/) (no git submodules) so a single download is self-contained for Zenodo / GitHub Release.

## Repository layout

| Directory | Role |
|-----------|------|
| [HydroModels_PBM](HydroModels_PBM/) | PBM calibration & evaluation (`XAJ-mz`) |
| [HydroModels_DLM](HydroModels_DLM/) | LSTM and physics-guided `dXAJ-mz` training |
| [HydroScirpt](HydroScirpt/) | Paper experiment scripts: `Sec0` HP sweep → `Sec1` baselines → `Sec2` PF-loss sensitivity |

## Quick start

```bash
# Python deps: see HydroModels_DLM/requirements.txt (and PBM env as needed)
export PYTHONPATH="$PWD/HydroModels_DLM:$PWD/HydroModels_PBM:$PYTHONPATH"

# Example — Sec0 Regional HP sweep
python HydroScirpt/Model/Sec0_ModelConfig/step1_SeqRegLSTM_Regional.py
```

Update `data_cfgs.input_path` in each script to your local Anhui processed dataset path. See [HydroScirpt/README.md](HydroScirpt/README.md) for the full section workflow.

## License

MIT License © 2025 Yikai CHAI
