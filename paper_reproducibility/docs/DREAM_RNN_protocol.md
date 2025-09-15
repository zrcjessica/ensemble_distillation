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

lentiMPRA (K562/HepG2, activity-only teachers or activity+aleatoric teachers per dataset):
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

## 2) Generate distillation targets

DeepSTARR:
- Ensemble mean: `paper_reproducibility/code/ensemble_predict_DeepSTARR.py --distill`
- Ensemble std: `paper_reproducibility/code/stdev_ensemble_predictions.py`
- Parse HDF5 with ensemble mean for train and ensemble std for train/val/test as per `DeepSTARR_ensemble_distillation_protocol.md`.

lentiMPRA:
- Ensemble mean (activity, and aleatoric if applicable): `paper_reproducibility/code/ensemble_predict_lentiMPRA.py --distill [--aleatoric]`
- Ensemble std (for activity): `paper_reproducibility/code/get_lentiMPRA_ensemble_std.py`
- Parse HDF5 containing updated labels for distilled runs as per `lentiMPRA_ensemble_distillation_protocol.md`.

## 3) Train distilled students (enable extra heads only for distillation)

DeepSTARR distilled:
- Student architecture outputs 4 heads (Dev, Hk, Dev_std, Hk_std) when `std: true` in YAML.
- Use the parsed HDF5 with ensemble labels.
- Script: `paper_reproducibility/code/train_degu_distilled_model_deepstarr.py` (existing)
- Ensure config flags: `std: true` (and keep epochs/batch/lr as above).

lentiMPRA distilled:
- Student architecture can output:
  - 2 heads (activity, epistemic): set `epistemic: true`, `aleatoric: false`
  - 3 heads (activity, aleatoric, epistemic): set both `aleatoric: true` and `epistemic: true`
- Use the parsed HDF5 with ensemble labels.
- Script options: `train_degu_distilled_model_lentimpra.py` or a DREAM-RNN-specific distilled runner if desired.

## Notes
- Standard models remain unaffected (no extra heads) unless `std/aleatoric/epistemic` flags are set in YAML.
- Hyperparameters are aligned with the DREAM paper (epochs=80, batch_size=1024, AdamW, lr=0.005, one-cycle LR).
- Outputs (weights, performance CSVs) follow the repository’s expected naming for interoperability with plotting.

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
