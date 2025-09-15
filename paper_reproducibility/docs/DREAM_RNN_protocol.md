# DREAM-RNN training and distillation guide (DeepSTARR and lentiMPRA)

This guide mirrors the style of `DeepSTARR_ensemble_distillation_protocol.md` with concise steps to train and distill DREAM-RNN for STARR-seq (DeepSTARR) and lentiMPRA datasets.

- Models are implemented in `paper_reproducibility/code/model_zoo.py`.
- Configs live in `paper_reproducibility/config/`.
- Outputs follow the same directory structure expected by plotting and analysis scripts.

## 1) Train teacher ensembles (standard)

DeepSTARR (Dev/Hk):
- Script: `paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py`
- Config: `paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml` (epochs=80, batch_size=1024, AdamW, lr=0.005)
- Heads: 2 (Dev, Hk). Ensure `std: false` in YAML.
- Example:
```
python paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py \
  --data <DeepSTARR_h5> \
  --config paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml \
  --downsample 1.0 --ix 0 --out <out_dir> --gpu 0
```

lentiMPRA (K562/HepG2):
- Script: `paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py`
- Config: `paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml`
- Heads (teacher):
  - Activity-only: set `aleatoric: false` in YAML (1 head)
  - Activity+aleatoric: set `aleatoric: true` in YAML (2 heads)
- Example:
```
python paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py \
  --data <lentiMPRA_h5> \
  --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
  --downsample 1.0 --ix 0 --out <out_dir> --gpu 0
```

## 2) Evaluate ensemble and generate distillation data

Use the DREAM-RNN ensemble evaluator (PyTorch) with repo-style flags. It writes:
- `ensemble_performance_avg.csv` if `--eval`
- `ensemble_avg_y_train.npy` if `--distill`
- Unified NPZ: `distillation_data_<DATASET>_<CELLTYPE>_<DOWN>.npz` with `train_mean/std` and `test_mean/std` for feeding distilled scripts.

Examples:
- DeepSTARR:
```
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset DeepSTARR --model_dir <teacher_dir> --n_mods 10 \
  --data <DeepSTARR_h5> --out <ensemble_out> --eval --distill --downsample 1.0
```
- lentiMPRA:
```
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset lentiMPRA --celltype K562 --aleatoric \
  --model_dir <teacher_dir> --n_mods 10 --data <lentiMPRA_h5> \
  --out <ensemble_out> --eval --distill --downsample 1.0
```

## 3) Train distilled students (enable extra heads only for distillation)

DeepSTARR distilled:
- Script: `paper_reproducibility/code/train_degu_distilled_model_deepstarr.py`
- Config: `paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml` with `std: true`
- Inputs: unified NPZ from step 2 and/or `ensemble_avg_y_train.npy`

lentiMPRA distilled:
- Script: `paper_reproducibility/code/train_degu_distilled_model_lentimpra.py`
- Config: `paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml` with `epistemic: true` (and `aleatoric: true` if 3-head student)
- Inputs: unified NPZ from step 2 and/or `ensemble_avg_y_train.npy`

## 4) Evaluation of distilled models

- Use existing evaluation patterns (saved `*_performance.csv`) and plotting scripts (e.g., `plot_ensemble_performance_comparison.py`)—outputs from steps 2–3 follow the repo naming conventions.

## Notes
- Standard models remain unaffected (no extra heads) unless `std/aleatoric/epistemic` flags are set in YAML.
- Hyperparameters are aligned with the DREAM paper (epochs=80, batch_size=1024, AdamW, lr=0.005, one-cycle LR).
- Outputs (weights, performance CSVs) integrate with existing analysis and plotting utilities.

## Minimal examples

DeepSTARR standard (teacher, 2 heads):
```
python paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py \
  --data zenodo/data/DeepSTARR_distillation_data.h5 \
  --config paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml \
  --ix 0 --out results/DREAM_RNN/DeepSTARR/full/standard --gpu 0
```

lentiMPRA standard (teacher, activity-only):
```
python paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py \
  --data zenodo/data/lentiMPRA_K562.h5 \
  --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
  --ix 0 --out results/DREAM_RNN/lentiMPRA/K562/full/standard --gpu 0
```

lentiMPRA distilled (activity + epistemic):
- Ensure HDF5 with ensemble labels is prepared per protocol.
- Set in YAML: `epistemic: true`, `aleatoric: false`.
