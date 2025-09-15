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
- `distillation_data_<DATASET>_<DOWNSAMPLE>.npz` (DeepSTARR) or `distillation_data_<DATASET>_<CELLTYPE>_<DOWNSAMPLE>.npz` (lentiMPRA) - contains:
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
  --distillation_data results/DREAM_RNN/DeepSTARR/ensemble/distillation_data_DeepSTARR_1.0.npz \
  --output_dir results/DREAM_RNN/DeepSTARR/distilled \
  --epochs 80 --batch_size 1024 --learning_rate 0.005
```

### lentiMPRA distilled:
- **Script**: `paper_reproducibility/code/train_degu_distilled_model_lentimpra.py`
- **Architecture**: DREAM-RNN with configurable outputs based on uncertainty type:
  - Epistemic only: `[activity, epistemic]` (2 outputs)
  - Aleatoric only: `[activity, aleatoric]` (2 outputs) 
  - Both: `[activity, aleatoric, epistemic]` (3 outputs)
- **Input**: NPZ file from step 2 containing ensemble predictions
- **Configuration**: Set `aleatoric: true/false` and `epistemic: true/false` in YAML config
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
- **Important**: Use the same downsample ratio across all pipeline steps for consistency

### Data requirements
- **DeepSTARR**: `zenodo/data/DeepSTARR_distillation_data.h5` with Train/Val/Test splits
- **lentiMPRA**: `zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5` or `lentiMPRA_HepG2_activity_and_aleatoric_data.h5`
- All data files should contain X (sequences) and y (labels) arrays
- Sequences are automatically transposed from (N, L, 4) to (N, 4, L) for PyTorch compatibility

### Portability and path handling
- **Relative path resolution**: All scripts use `Path(__file__).resolve().parent.parent.parent` to find the repository root
- **Cross-platform compatibility**: Scripts work regardless of execution location or operating system
- **Automatic dependency resolution**: Scripts automatically add the code directory to Python path
- **Data file discovery**: Scripts can find data files in `zenodo/data/` relative to repository root
- **No hardcoded paths**: All paths are resolved relative to script location for maximum portability

### Architecture consistency
- **Standard models**: 2 outputs (DeepSTARR: Dev/Hk, lentiMPRA: activity)
- **Distilled models**: 4 outputs (DeepSTARR: Dev/Hk + std) or 3 outputs (lentiMPRA: activity + aleatoric + epistemic)
- **Full DREAM-RNN**: Bi-LSTM layers with convolutional backbone as per @DREAM_paper

### Modular architecture design
The DREAM-RNN implementation follows a modular approach that allows flexible uncertainty estimation:

- **Base models**: Standard DREAM-RNN with 2 outputs (DeepSTARR: Dev/Hk, lentiMPRA: activity)
- **Distillation modification**: Distillation scripts modify the final layer to add uncertainty heads
  - DeepSTARR: 2 outputs → 4 outputs (Dev, Hk, Dev_std, Hk_std)
  - lentiMPRA: 1 output → 3 outputs (activity, aleatoric, epistemic)
- **Implementation**: Uses `nn.ModuleDict` for final layer to allow easy modification by distillation scripts

### Uncertainty estimation options (per @DEGU_paper)
Following the DEGU methodology, different uncertainty types can be estimated:

**Epistemic uncertainty** (model uncertainty):
- Captured by variability across ensemble predictions
- Available for both DeepSTARR and lentiMPRA
- Represents uncertainty due to limited training data

**Aleatoric uncertainty** (data uncertainty):
- Captured by variability across experimental replicates
- Available for lentiMPRA (3 replicates) and ATAC-seq (3 replicates)
- Represents irreducible noise in experimental measurements

**Combined uncertainty**:
- Total uncertainty = √(epistemic² + aleatoric²)
- Provides comprehensive uncertainty assessment
- Enables better calibration and decision-making

**Configuration options**:
- Standard training: No uncertainty heads
- Epistemic only: Add epistemic uncertainty head (ensemble variability)
- Aleatoric only: Add aleatoric uncertainty head (replicate variability) 
- Both: Add both epistemic and aleatoric uncertainty heads

### Hyperparameters
Aligned with @DREAM_paper specifications:
- **Epochs**: 80 (configurable via YAML)
- **Batch size**: 1024 (configurable via YAML)  
- **Optimizer**: AdamW with lr=0.005
- **Scheduler**: OneCycleLR
- **Architecture**: Full DREAM-RNN with Bi-LSTM layers
- **Dropout**: 0.2
- **LSTM hidden dimensions**: 320 (640 total with bidirectional)
- **Kernel sizes**: [9, 15] for first convolutional layer

### Command line arguments
All scripts support standard arguments:
- `--data`: Path to HDF5 data file
- `--out`/`--output_dir`: Output directory for models and results
- `--config`: Path to YAML configuration file
- `--downsample`: Downsample ratio (0.0-1.0)
- `--ix`: Model index for ensemble training
- `--gpu`: GPU device ID (0, 1, 2, etc.)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training

### Scripts overview
- `train_DREAM_RNN_DeepSTARR.py`: Standard DREAM-RNN training for DeepSTARR
- `train_DREAM_RNN_lentiMPRA.py`: Standard DREAM-RNN training for lentiMPRA
- `evaluate_ensemble_and_generate_distillation_data.py`: Ensemble evaluation and distillation data generation
- `train_degu_distilled_model_deepstarr.py`: Distilled model training for DeepSTARR
- `train_degu_distilled_model_lentimpra.py`: Distilled model training for lentiMPRA

## Configuration examples for different uncertainty types

### DeepSTARR uncertainty configurations:
DeepSTARR distillation always includes epistemic uncertainty (ensemble variability):

```yaml
# DREAM_RNN_DeepSTARR.yaml for distilled models
epochs: 80
batch_size: 1024
optimizer: 'AdamW'
optim_lr: 0.005
std: true  # Enable epistemic uncertainty (ensemble variability)
```

### lentiMPRA uncertainty configurations:

**Epistemic uncertainty only** (ensemble variability):
```yaml
# DREAM_RNN_lentiMPRA.yaml
epochs: 80
batch_size: 1024
optimizer: 'AdamW'
optim_lr: 0.005
aleatoric: false  # No aleatoric uncertainty
epistemic: true   # Enable epistemic uncertainty
```

**Aleatoric uncertainty only** (experimental replicates):
```yaml
# DREAM_RNN_lentiMPRA.yaml
epochs: 80
batch_size: 1024
optimizer: 'AdamW'
optim_lr: 0.005
aleatoric: true   # Enable aleatoric uncertainty
epistemic: false  # No epistemic uncertainty
```

**Both uncertainties** (recommended for full DEGU methodology):
```yaml
# DREAM_RNN_lentiMPRA.yaml
epochs: 80
batch_size: 1024
optimizer: 'AdamW'
optim_lr: 0.005
aleatoric: true   # Enable aleatoric uncertainty
epistemic: true   # Enable epistemic uncertainty
```

## Complete workflow example

### DeepSTARR workflow:
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
    --distillation_data results/DREAM_RNN/DeepSTARR/ensemble/distillation_data_DeepSTARR_1.0.npz \
    --output_dir results/DREAM_RNN/DeepSTARR/distilled \
    --model_index $i \
    --epochs 80 --batch_size 1024
done
```

### lentiMPRA workflow (K562):
```bash
# 1. Train 10 standard models (lentiMPRA K562)
for i in {0..9}; do
  python paper_reproducibility/code/train_DREAM_RNN_lentiMPRA.py \
    --data zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/DREAM_RNN/lentiMPRA/K562/standard \
    --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
    --ix $i --gpu 0
done

# 2. Evaluate ensemble and generate distillation data
python paper_reproducibility/code/evaluate_ensemble_and_generate_distillation_data.py \
  --dataset lentiMPRA \
  --celltype K562 \
  --model_dir results/DREAM_RNN/lentiMPRA/K562/standard \
  --n_mods 10 \
  --data zenodo/data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
  --out results/DREAM_RNN/lentiMPRA/K562/ensemble \
  --eval --distill --downsample 1.0

# 3. Train 5 distilled models
for i in {0..4}; do
  python paper_reproducibility/code/train_degu_distilled_model_lentimpra.py \
    --distillation_data results/DREAM_RNN/lentiMPRA/K562/ensemble/distillation_data_lentiMPRA_K562_1.0.npz \
    --output_dir results/DREAM_RNN/lentiMPRA/K562/distilled \
    --config paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml \
    --model_index $i \
    --epochs 80 --batch_size 1024
done
```

## Troubleshooting

### Common issues:
1. **CUDA out of memory**: Reduce batch size in config file or use smaller downsample ratio
2. **File not found errors**: Ensure data files are in `zenodo/data/` directory relative to repository root
3. **Model loading errors**: Check that model files exist in the specified directory
4. **Import errors**: Ensure all dependencies are installed (PyTorch, h5py, numpy, etc.)
5. **Path resolution errors**: Scripts automatically resolve paths relative to their location - no need to run from specific directories

### Performance tips:
- Use GPU acceleration with `--gpu 0` (or other available GPU)
- For testing, use small downsample ratios (0.01, 0.1) to speed up training
- Monitor GPU memory usage and adjust batch size accordingly
- For quick validation runs, create temporary config files with reduced epochs and batch sizes
