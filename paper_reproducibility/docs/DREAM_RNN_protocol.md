# DREAM-RNN training and distillation guide (DeepSTARR and lentiMPRA)

This guide provides step-by-step instructions for training and distilling DREAM-RNN models for STARR-seq (DeepSTARR) and lentiMPRA datasets using the ensemble distillation pipeline.

- **Models**: PyTorch implementations in `paper_reproducibility/code/` (with Keras backbones in `model_zoo.py`)
- **Configs**: YAML files in `paper_reproducibility/config/`
- **Data**: HDF5 files in `zenodo/data/`
- **Outputs**: Follow the same directory structure expected by plotting and analysis scripts

## 1) Train teacher ensembles (standard)

### DeepSTARR (Dev/Hk):
- **Script**: `paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py`
- **Config**: `paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml` (epochs=80, batch_size=1024, AdamW, lr=0.005)
- **Architecture**: Full DREAM-RNN with Bi-LSTM layers
- **Outputs**: 2 heads (Dev, Hk) - ensure `std: false` in YAML for standard models
- **Data**: `zenodo/data/DeepSTARR_distillation_data.h5`
- **Example**:
```bash
python paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py \
  --data zenodo/data/DeepSTARR_distillation_data.h5 \
  --out results/DREAM_RNN/DeepSTARR/standard \
  --config paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml \
  --downsample 1.0 --ix 0 --gpu 0
```

### lentiMPRA (K562/HepG2):
- **Script**: `paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py`
- **Config**: `paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml`
- **Architecture**: Full DREAM-RNN with Bi-LSTM layers
- **Outputs**: 
  - Activity-only: set `aleatoric: false` in YAML (1 head)
  - Activity+aleatoric: set `aleatoric: true` in YAML (2 heads)
- **Data**: `zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5` or `lentiMPRA_HepG2_activity_and_aleatoric_data.h5`
- **Example**:
```bash
python paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py \
  --data zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
  --out results/DREAM_RNN/lentiMPRA/K562/standard \
  --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
  --downsample 1.0 --ix 0 --gpu 0
```

## 2) Evaluate ensemble and generate distillation data

The ensemble evaluator loads multiple trained models, generates predictions, and creates distillation data for training student models.

### Outputs:
- `ensemble_performance_avg.csv` - ensemble performance metrics (if `--eval`)
- `distillation_data_<DATASET>_<CELLTYPE>_<DOWNSAMPLE>.npz` - contains:
  - `train_mean`, `train_std` - ensemble predictions for training data
  - `val_mean`, `val_std` - ensemble predictions for validation data  
  - `test_mean`, `test_std` - ensemble predictions for test data

### DeepSTARR:
```bash
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset DeepSTARR \
  --model_dir results/DREAM_RNN/DeepSTARR/standard \
  --n_mods 10 \
  --data zenodo/data/DeepSTARR_distillation_data.h5 \
  --out results/DREAM_RNN/DeepSTARR/ensemble \
  --eval --distill --downsample 1.0
```

### lentiMPRA:
```bash
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset lentiMPRA \
  --celltype K562 \
  --model_dir results/DREAM_RNN/lentiMPRA/K562/standard \
  --n_mods 10 \
  --data zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
  --out results/DREAM_RNN/lentiMPRA/K562/ensemble \
  --eval --distill --downsample 1.0
```

## 3) Train distilled students

Distilled models learn to approximate ensemble predictions and provide uncertainty estimates.

### DeepSTARR distilled:
- **Script**: `paper_reproducibility/code/train_degu_distilled_model_deepstarr.py`
- **Architecture**: DREAM-RNN with 4 outputs: `[Dev_mean, Hk_mean, Dev_std, Hk_std]`
- **Input**: NPZ file from step 2 containing ensemble predictions
- **Example**:
```bash
python paper_reproducibility/code/train_degu_distilled_model_deepstarr.py \
  --distillation_data results/DREAM_RNN/DeepSTARR/ensemble/distillation_data_DeepSTARR_Dev_1.0.npz \
  --output_dir results/DREAM_RNN/DeepSTARR/distilled \
  --epochs 80 --batch_size 1024 --learning_rate 0.005
```

### lentiMPRA distilled:
- **Script**: `paper_reproducibility/code/train_degu_distilled_model_lentimpra.py`
- **Architecture**: DREAM-RNN with 3 outputs: `[activity, aleatoric, epistemic]`
- **Input**: NPZ file from step 2 containing ensemble predictions
- **Example**:
```bash
python paper_reproducibility/code/train_degu_distilled_model_lentimpra.py \
  --distillation_data results/DREAM_RNN/lentiMPRA/K562/ensemble/distillation_data_lentiMPRA_K562_1.0.npz \
  --output_dir results/DREAM_RNN/lentiMPRA/K562/distilled \
  --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
  --epochs 80 --batch_size 1024
```

## 4) Evaluation and analysis

All scripts generate performance CSV files that integrate with existing analysis and plotting utilities:
- Standard models: `*_performance.csv` with Pearson/Spearman correlations and MSE
- Ensemble evaluation: `ensemble_performance_avg.csv` with ensemble metrics
- Distilled models: `*_performance.csv` with uncertainty-aware performance metrics

Use existing plotting scripts (e.g., `plot_ensemble_performance_comparison.py`) for visualization.

## Key features

### On-the-fly downsampling
All scripts support on-the-fly downsampling with fixed seed (1234) for reproducibility:
- Use `--downsample 0.1` for 10% of data, `--downsample 0.005` for 0.5%, etc.
- Training data is downsampled while validation/test data remains full size
- Consistent across standard training, ensemble evaluation, and distillation

### Architecture consistency
- **Standard models**: 2 outputs (DeepSTARR: Dev/Hk, lentiMPRA: activity)
- **Distilled models**: 4 outputs (DeepSTARR: Dev/Hk + std) or 3 outputs (lentiMPRA: activity + aleatoric + epistemic)
- **Full DREAM-RNN**: Bi-LSTM layers with convolutional backbone as per @DREAM_paper

### Hyperparameters
Aligned with @DREAM_paper specifications:
- **Epochs**: 80 (configurable via YAML)
- **Batch size**: 1024 (configurable via YAML)  
- **Optimizer**: AdamW with lr=0.005
- **Scheduler**: OneCycleLR
- **Architecture**: Full DREAM-RNN with Bi-LSTM layers

## Complete workflow example

```bash
# 1. Train 10 standard models (DeepSTARR)
for i in {0..9}; do
  python paper_reproducibility/code/train_DREAM_RNN_DeepSTARR.py \
    --data zenodo/data/DeepSTARR_distillation_data.h5 \
    --out results/DREAM_RNN/DeepSTARR/standard \
    --config paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml \
    --ix $i --gpu 0
done

# 2. Evaluate ensemble and generate distillation data
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset DeepSTARR \
  --model_dir results/DREAM_RNN/DeepSTARR/standard \
  --n_mods 10 \
  --data zenodo/data/DeepSTARR_distillation_data.h5 \
  --out results/DREAM_RNN/DeepSTARR/ensemble \
  --eval --distill --downsample 1.0

# 3. Train 5 distilled models
for i in {0..4}; do
  python paper_reproducibility/code/train_degu_distilled_model_deepstarr.py \
    --distillation_data results/DREAM_RNN/DeepSTARR/ensemble/distillation_data_DeepSTARR_Dev_1.0.npz \
    --output_dir results/DREAM_RNN/DeepSTARR/distilled \
    --model_index $i \
    --epochs 80 --batch_size 1024
done
```
