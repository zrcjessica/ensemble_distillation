# DEGU: Distilling Ensembles for Genomic Uncertainty-aware Models

A lightweight Python framework for ensemble learning and knowledge distillation in genomic deep learning, providing uncertainty quantification and improved model robustness.

## Overview

DEGU implements a two-stage approach:
1. **Ensemble Training**: Train multiple models with different weight initializations
2. **Knowledge Distillation**: Transfer ensemble knowledge to a single efficient student model

The framework provides both epistemic uncertainty (model disagreement) and enables aleatoric uncertainty modeling (data variability) for robust genomic predictions.

## Key Features

- **Lightweight Implementation**: Simple, extensible code that works with any Keras model
- **Uncertainty Quantification**: Epistemic uncertainty from ensemble disagreement
- **Knowledge Distillation**: Compress ensemble knowledge into a single model
- **Improved Attributions**: More stable feature importance analysis
- **Enhanced Generalization**: Better performance on out-of-distribution data

## Installation

```bash
pip install tensorflow numpy scipy scikit-learn h5py
```


## Examples

- **Complete Example Script**: `degu_example_deepstarr.py` - Full working example with the DeepSTARR model
- **Interactive Tutorial**: [Google Colab Notebook](https://colab.research.google.com/drive/1XE8ATbwUXu-g8dTuxNX5GM2K_Lhe7V7R?usp=sharing) - Try DEGU in your browser


## Quick Start

Here is an example for how to customize the code for DeepSTARR:
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from degu import DEGU, uncertainty_logvar, standard_train_fun, eval_regression

# Define your model architecture
def DeepSTARR(input_shape, output_shape):
    """DeepSTARR model from deAlmeida et al. Nat Genetics (2022)"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(60, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(60, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(120, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(output_shape, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

# Load your data
# x_train, y_train, x_valid, y_valid, x_test, y_test = load_your_data()

# Get data dimensions
N, L, A = x_train.shape
num_targets = y_train.shape[1]

# Step 1: Train ensemble
base_model = DeepSTARR(input_shape=(L,A), output_shape=num_targets)
degu = DEGU(base_model, num_ensemble=10, uncertainty_fun=uncertainty_logvar)

optimizer = keras.optimizers.Adam(learning_rate=0.002)
history = degu.train_ensemble(
    x_train, y_train,
    train_fun=standard_train_fun,
    save_prefix='model_ensemble',
    optimizer=optimizer,
    loss='mse',
    validation_data=(x_valid, y_valid)
)

# Step 2: Evaluate ensemble
results = degu.eval_ensemble(x_test, y_test, eval_fun=eval_regression)
ensemble_results, standard_results, predictions, uncertainties = results

# Step 3: Distill to student model (2x outputs for predictions + uncertainties)
student_model = DeepSTARR(input_shape=(L,A), output_shape=num_targets*2)
student_model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse')

history = degu.distill_student(
    student_model, x_train, y_train,
    train_fun=standard_train_fun,
    save_prefix='model_distilled',
    validation_data=(x_valid, y_valid),
    batch_size=128
)

# Step 4: Evaluate student model
student_results, student_pred, y_ensemble = degu.eval_student(
    student_model, x_test, y_test, eval_fun=eval_regression
)
```

## Key Components

### DEGU Class
- `train_ensemble()`: Train ensemble with different weight initializations
- `pred_ensemble()`: Generate predictions with uncertainty estimates
- `distill_student()`: Train student model using ensemble knowledge
- `eval_ensemble()`: Comprehensive ensemble evaluation
- `eval_student()`: Student model evaluation against ensemble

### Uncertainty Functions
- `uncertainty_logvar()`: Log variance uncertainty (default)
- `uncertainty_std()`: Standard deviation uncertainty

### Training Functions
- `standard_train_fun()`: Standard training with early stopping and LR scheduling

### Evaluation Functions
- `eval_regression()`: MSE, Pearson, Spearman correlations
- `eval_classification()`: AUROC, AUPR, F1-score

## Model Architecture Requirements

For knowledge distillation, your student model should have **2x the output dimensions** of the base model to predict both values and uncertainties:

```python
# Base model: predicts values only
base_model = YourModel(output_shape=num_targets)

# Student model: predicts values + uncertainties  
student_model = YourModel(output_shape=num_targets * 2)
```

## Extending to Other Models

DEGU works with any Keras model. Simply:

1. Define your model architecture function
2. Ensure proper input/output shapes
3. Use DEGU's training and distillation pipeline

```python
def YourCustomModel(input_shape, output_shape):
    # Your model architecture here
    return keras.Model(inputs=inputs, outputs=outputs)

# Use with DEGU
base_model = YourCustomModel(input_shape=your_input_shape, output_shape=your_output_shape)
degu = DEGU(base_model, num_ensemble=5)
# ... rest of pipeline
```

## Paper and Citation

This implementation accompanies our **ICLR MLGenX 2025** paper:

**"Uncertainty-Aware Genomic Deep Learning with Knowledge Distillation"**  
*Jessica Zhou, Kaeli Rizzo, Trevor Christensen, Ziqi (Amber) Tang, Peter K Koo*

```bibtex
@article{Zhou2024.11.13.623485,
    author = {Zhou, Jessica and Rizzo, Kaeli and Christensen, Trevor and Tang, Ziqi and Koo, Peter K},
    title = {Uncertainty-aware genomic deep learning with knowledge distillation},
    year = {2024},
    doi = {10.1101/2024.11.13.623485},
    publisher = {Cold Spring Harbor Laboratory},
    journal = {bioRxiv}
}
```

## Repository Structure

```
├── degu.py                   # Core DEGU implementation
├── paper_reproducibility/    # Reproduction code and configs
│   ├── code/                 # Analysis scripts  
│   └── config/               # Configuration files
│   ├── data_preprocessing/   # Data processing code and notebooks
│   └── docs/                 # Overview of each experiment and guidance for code
└── README.md                 # This file
```

## Reproducibility

Full reproduction code for paper results is available in `paper_reproducibility/`. This includes dataset preprocessing, model training scripts, analysis notebooks, and documentation for all experiments in the paper.

## Data and Model Availability

Pre-trained model weights and datasets used in our experiments are available on Zenodo:

**DOI:** 10.5281/zenodo.14145284  
**Link:** https://zenodo.org/records/14145285

This includes:
- Trained ensemble model weights for all experiments
- Distilled student model weights
- Processed datasets for DeepSTARR, lentiMPRA, and ATAC-seq experiments

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact koo@cshl.edu.