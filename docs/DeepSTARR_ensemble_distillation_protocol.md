# DeepSTARR ensemble distillation workflow and code
- [DeepSTARR ensemble distillation workflow and code](#deepstarr-ensemble-distillation-workflow-and-code)
- [1. Train ensemble of DeepSTARR models with standard training](#1-train-ensemble-of-deepstarr-models-with-standard-training)
  - [Code](#code)
  - [Outputs](#outputs)
    - [Standard training](#standard-training)
    - [Training with EvoAug](#training-with-evoaug)
- [2. Generate training data for distilled models](#2-generate-training-data-for-distilled-models)
  - [2a. Calculate ensemble average](#2a-calculate-ensemble-average)
    - [Code](#code-1)
    - [Outputs](#outputs-1)
  - [2b. Calculate ensemble standard deviation](#2b-calculate-ensemble-standard-deviation)
    - [Code](#code-2)
    - [Outputs](#outputs-2)
  - [2c. Parse HDF5 file with data for training distilled models](#2c-parse-hdf5-file-with-data-for-training-distilled-models)
    - [Code](#code-3)
    - [Output](#output)
- [3. Train distilled models with epistemic uncertainty estimates](#3-train-distilled-models-with-epistemic-uncertainty-estimates)
  - [Code](#code-4)
  - [Outputs](#outputs-3)
    - [With EvoAug](#with-evoaug)
- [Attribution analysis](#attribution-analysis)
  - [1. Calculate attribution maps](#1-calculate-attribution-maps)
    - [Code](#code-5)
    - [Outputs](#outputs-4)
      - [1. Model-Specific Attribution Maps:](#1-model-specific-attribution-maps)
      - [2. Average Attribution Map Across Ensemble (if `--average` flag is set):](#2-average-attribution-map-across-ensemble-if---average-flag-is-set)
  - [2. Analyze attribution scores](#2-analyze-attribution-scores)
    - [Code](#code-6)
    - [Outputs](#outputs-5)
      - [1. RMSE Calculation (if `--rmse` flag is set)](#1-rmse-calculation-if---rmse-flag-is-set)
      - [2. Variance Calculation (if `--var` flag is set)](#2-variance-calculation-if---var-flag-is-set)
  - [3. Visualizing results](#3-visualizing-results)
- [Generating shifted sequences](#generating-shifted-sequences)
  - [Code](#code-7)
- [Training with dynamic augmentations](#training-with-dynamic-augmentations)
  - [Code](#code-8)
  - [Outputs](#outputs-6)
- [Evaluating different ensemble sizes](#evaluating-different-ensemble-sizes)
  - [1. Train up to $N$ teacher models with standard training](#1-train-up-to-n-teacher-models-with-standard-training)
    - [Code](#code-9)
    - [Outputs](#outputs-7)
  - [2. Evaluate performance of different ensemble sizes and get training data for distilled models](#2-evaluate-performance-of-different-ensemble-sizes-and-get-training-data-for-distilled-models)
    - [Code](#code-10)
    - [Outputs](#outputs-8)
  - [3. Train distilled models based on different teacher ensemble sizes](#3-train-distilled-models-based-on-different-teacher-ensemble-sizes)
    - [Code](#code-11)
    - [Outputs](#outputs-9)

Note: all bash scripts use a Slack bot to message user upon completion. To use this feature, refer to the Slack docs: [Sending messages using incoming webhooks](https://api.slack.com/messaging/webhooks)

# 1. Train ensemble of DeepSTARR models with standard training
Train an ensemble of DeepSTARR models with standard training. These models serve as teacher models for DEGU distillation.

## Code
- `train_DeepSTARR.py`: train an ensemble of DeepSTARR models with standard training; takes a config (`DeepSTARR.yaml`) file as one of the input arguments. Set `--evoaug` flag to train models with EvoAug. Set `--downsample` flag and provide downsampling proportions to train models on random subsets of the training data. Other input arguments defined in `parse_args()` function. 
- `train_DeepSTARR_ensemble.sh`: wrapper script for running `train_DeepSTARR.py` on GPUs. Must provide relevant input arguments to script and define options such as EvoAug and downsampling. 

## Outputs

### Standard training 
- `<model_ix>_DeepSTARR.h5`: Trained DeepSTARR model (saved w/ `model.save()`). 
- `<model_ix>_historyDict`: Pickled model history (from `history.history`).
- `<model_ix>_performance.csv`: DeepSTARR model performance on held out test set.

Outputs for downsampled models are saved in a subdirectory named according to downsampling proportion (as defined in bash script).

### Training with EvoAug
- `<model_ix>_DeepSTARR_aug_weights.h5`: Weights of model trained w/ EvoAug (saved w/ `model.save_weights()`)
- `<model_ix>_historyDict_aug`: Pickled weights of augmented model.
- `<model_ix>_DeepSTARR_finetune.h5`: Weights of finetuned model (saved w/ `model.save_weights()`).
- `<model_ix>_historyDict_finetune`: Pickled history of finetuned model. 
- `<model_ix>_performance_finetune.csv`: Performance of finetuned model on test set.
  
# 2. Generate training data for distilled models 

## 2a. Calculate ensemble average 
Use the teacher ensemble to make predictions on training sequences and calculate the ensemble average. If using models trained on a subset of the training dataset, make predictions on the same subset of training data that the teacher model was trained on. The ensemble average is used as target labels for the training sequences for training the distilled models. 

### Code
- `ensemble_predict_DeepSTARR.py`: Uses a defined ensemble of DeepSTARR teacher models to perform inference on STARR-seq sequences and calculate the average of the ensemble's predictions. With `--distill` flag is set, inference is performed on training sequences and the ensemble average is saved as a numpy array to a `.npy`. If `--eval` flag is set, inference is performed on test sequences and performance of ensemble average is evaluated and saved to an output file. 
- `eval_and_distill_DeepSTARR.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set
  
### Outputs
- `ensemble_avg_y_train.npy`: Ensemble average predictions on train sequences (if `--distill` flag set).
- `ensemble_performance_avg.csv`: Ensemble average performance on test sequences (if `--eval` flag set).

## 2b. Calculate ensemble standard deviation
Use the teacher ensemble to make predictions on all STARR-seq data (train/val/test) and calculate the ensemble standard deviation of these performances. These values are used as new target labels for training the epistemic uncertainty output head of distilled DeepSTARR models. 

### Code
- `stdev_ensemble_predictions.py`: make predictions on train/test/val sequences with ensemble of DeepSTARR teacher models and save ensemble standard deviations as numpy arrays to `npy` files. 
- `generate_stdev_dataset.sh`: run the above script
### Outputs
- `ensemble_std_train.npy`: Ensemble standard deviation of predictions on train sequences.
- `ensemble_std_test.npy`: Ensemble standard deviation of predictions on test sequences.
- `ensemble_std_val.npy`: Ensemble standard deviation of predictions on val sequences. 
<!-- # 3. Evaluate model performance 
- for original models
  - trained on all training data
  - trained on subsets of training data
- for ensemble average
  - for ensembles trained on all training data
  - for ensembles trained on subsets of training data
- for distilled models 
  - trained on all training data
  - trained on subsets of training data
## Code
- `ensemble_predict_DeepSTARR.py` with `--eval` flag 
- `evaluate_ensemble_predictions.sh`
- `eval_and_distill_DeepSTARR.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set (prototype script, not used yet)
- `run_ensemble_predict_DeepSTARR_evoaug.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set for DeepSTARR ensembles trained w/ EvoAug
## Outputs
- `ensemble_performance_avg.csv`
- `<model_ix>_pred_scatterplot.png`: scatterplot of predicted vs. true values for test set
- `avg_pred_scatterplot.png` if `--plot` flag set -->

## 2c. Parse HDF5 file with data for training distilled models 
For all folds of data (train/test/val), add target labels for epistemic uncertainty obtained from step 2b. For training sequences, update target labels for sequence activity with ensemble average obtained from step 2a. Updated target labels have shape `(N,4)`. 

### Code
- `parse_DeepSTARR_data.ipynb`: located in `data_preprocessing` directory; requires outputs of steps 2a-b. You need to generate a separate HDF5 file corresponding to each downsampling proportion 
### Output
- `all_data_with_ensemble_metrics_hierarchical.h5`: Contains the same input sequences but with updated target labels for distillation. 
  
# 3. Train distilled models with epistemic uncertainty estimates 
Train a user-defined amount of distilled DeepSTARR models that predict Dev and Hk sequence activity and corresponding epistemic uncertainty. Requires the output of step 2. 

## Code
- `train_stdev_DeepSTARR.py`: Trains distilled model that predicts epistemic uncertainty (modified from `train_DeepSTARR.py`).
- `distill_DeepSTARR_with_std.sh`: Run `train_stdev_DeepSTARR.py`; toggle `evoaug` and `downsample` boolean variables accordingly.
- `distill_DeepSTARR_with_logvar.sh`: Runs `train_stdev_DeepSTARR.py` w/ `--logvar` flag. In this case, epistemic uncertainty is learned as log variance instead of standard deviation.

## Outputs
- `<model_ix>_DeepSTARR.h5`: Distilled DeepSTARR model (saved with `model.save()`).
- `<model_ix>_historyDict`: Pickled model history (from `history.history`).
- `<model_ix>_performance.csv`: Model performance on held out test sequences.

### With EvoAug
- `<model_ix>_DeepSTARR_aug_weights.h5`: Weights of model trained w/ EvoAug (saved w/ `model.save_weights()`).
- `<model_ix>_historyDict_aug`: Pickled weights of augmented model.
- `<model_ix>_DeepSTARR_finetune.h5`: Weights of finetuned model (saved w/ `model.save_weights()`).
- `<model_ix>_historyDict_finetune`: Pickled history of finetuned model.
- `<model_ix>_performance_finetune.csv`: Performance of finetuned model on test set.
<!-- - `<model_ix>_loss_curves.png` if run with `--plot` flag -->
<!-- # 3. Train distilled models 

- trained on all training data
- trained on subsets of training data, specifically using the same samples in the subsets used to train the downsampled ensembles... the random seed set in line 15 of `utils.py` *should* account for this. 

Requires output of step 2 (`ensemble_avg_y_train.npy`).

## Code
- `train_DeepSTARR.py`: with `--distill` flag set to `ensemble_avg_y_train.py` corresponding to training data size (full or downsampled)
- `train_DeepSTARR_distilled.sh`: for training distilled models on entire training dataset
- `train_DeepSTARR_distilled_downsampled.sh`: above, for downsampled training data
- `train_DeepSTARR_distilled_downsample_one_case.sh`: somewhat temporary script created to train downsampled models at p=0.75 because for some reason it was skipped
  
## Outputs
Same as [outputs of `train_DeepSTARR.py`](#outputs) but should be saved in a directory corresponding to ensemble distilled models. 

# 5. Evaluate performance of distilled models on test set 
- for distilled models trained on full training dataset
- for downsampled distilled models trained on subset of training data

## Code
- `ensemble_predict_DeepSTARR.py` with `--eval` flag (from [step 3](#Code-2))
- `eval_ensemble_preds_distilled.sh`: bash script specific for distilled models -->

<!-- # 8. Get ensemble standard deviation 
Get the ensemble standard deviation for all data (train/test/val) to train & evaluate distilled models with uncertainty estimation. 

## Code
- `stdev_ensemble_predictions.py`: make predictions on train/test/val data with DeepSTARR ensemble and save ensemble standard deviation to file
- `generate_stdev_dataset.sh`: run the above script

# 9. Make an h5 file that contains the DeepSTARR training data and the ensemble standard deviation 
This will be used to train distilled models with uncertainty estimation.

## Code
- `parse_DeepSTARR_data.ipynb`: (in `data_preprocessing` dir)

## Outputs
Currently using file `all_data_with_ensemble_metrics_hierarchicalHDF5`. -->

<!-- # 10. Train distilled model with stdev. prediction
We replace the original target labels with the average and standard deviation of the ensemble's predictions on the training sequences. For the validation and test sets, we keep the original enhancer activity values but append the standard deviation of the ensemble's predictions on those sequences. 

## Code
- `train_stdev_DeepSTARR.py`: python script for training distilled model w/ stdev. prediction (modified from `train_DeepSTARR.py`)
- `distill_DeepSTARR_with_std.sh`: run `train_stdev_DeepSTARR.py` 
  - can toggle `evoaug` and `downsample` boolean variables accordingly
- `distill_evoaug_DeepSTARR_with_std.sh`: run `train_stdev_DeepSTARR.py` with `--evoaug` flag (has now been wrapped into above script)
- `distill_DeepSTARR_with_logvar.sh`: runs `train_stdev_DeepSTARR.py` w/ `--logvar` flag (currently exploratory, in .gitignore) -->


<!-- # 11. Attribution analysis for DeepSTARR models with mean + std (epistemic) predictions 

## Code
- `attr_analysis_distilled_with_std.sh`: run DeepSTARR 
- `binned_attr_analysis_distilled_with_std.sh`: used to perform attribution analysis on 100 seqs randomly selected from 4 quantile bins based on ground truth Dev enhancer activity; runs `DeepSTARR_attr_analysis_for_binned_seqs.py` -->

# Attribution analysis
Calculate and analyze attribution maps for Dev and Hk output heads of DeepSTARR models with standard training and DEGU-distillation using saliency maps and DeepSHAP. 

## 1. Calculate attribution maps 
For each activity output head, calculate attribution maps w.r.t. 1000 test sequences with the largest experimental activity measurements. 
### Code
- `DeepSTARR_ensemble_attr_analysis.py`: This script performs attribution analysis (either saliency or DeepSHAP) on the top `top_n` predictions from a test set for an ensemble of DeepSTARR models (or distilled DeepSTARR models). It can analyze predictions for a specific promoter class (Dev/Hk) and output head (mean/std/logvar) and calculates attribution scores for each model in the set of models provided. Can optionally calculate the average attribution map across the ensemble of models if `--average` flag set.
- `run_DeepSTARR_ensemble_attr_analysis.sh`: runs `DeepSTARR_ensemble_attr_analysis.py` to perform attribution analysis on DeepSTARR models trained with full dataset
### Outputs
#### 1. Model-Specific Attribution Maps:
For each model in the ensemble, the script saves an attribution map (gradient) of the top n examples (e.g., top 500 predictions). Each file follows this naming convention:
- If `--method=shap` and `--dinuc_shuffle` is set:
  ```
  <model_index>_top<top_n>_<enhancer>-<head>_shap_dinuc_shuffle.npy
  ```
- If `--method=shap` without `--dinuc_shuffle`:
  ```
  <model_index>_top<top_n>_<enhancer>-<head>_shap.npy
  ```
- If `--method=saliency`:
  ```
  <model_index>_top<top_n>_<enhancer>-<head>_saliency.npy
  ```
#### 2. Average Attribution Map Across Ensemble (if `--average` flag is set):
The script computes and saves an average attribution map across all models. This file is useful for visualizing an aggregate attribution pattern across models.
- If `--dinuc_shuffle` is set with `--method=shap`:
  ```
  avg_top<top_n>_<enhancer>-<head>_shap_dinuc_shuffle.npy
  ```
- If only `--method=shap` is specified:
  ```
  avg_top<top_n>_<enhancer>-<head>_shap.npy
  ```
- If `--method=saliency` is specified:
  ```
  avg_top<top_n>_<enhancer>-<head>_saliency.npy
  ```
Each .npy file contains an array of attribution scores, representing either individual model outputs or the averaged results across models. These outputs can be analyzed further to understand feature importance for top predictions in the test set.

  <!-- - `run_DeepSTARR_ensemble_attr_analysis_downsampled.sh`: runs attribution analysis on DeepSTARR models trained w/ downsampled training dataset. Focuses on Dev enhancer activity output head. -->
  <!-- - `run_evoaug_DeepSTARR_ensemble_attr_analysis.sh`: runs attribution analysis on DeepSTARR models trained w/ evoaug on full dataset; has been modified to include boolean flags for downsampled and distilled models -->

## 2. Analyze attribution scores
We perform two analyses of the attribution maps obtained in step [1](#1-calculate-attribution-maps):
1. RMSE from ensemble average attribution score
2. Standard deviation of attribution scores
### Code
- `analyze_attr_scores.py`: This script performs analysis on attribution scores generated from an ensemble of DeepSTARR models (or distilled DeepSTARR models). It supports two types of attribution analysis methods (saliency and SHAP), and calculates either the root mean squared error (RMSE) between each model’s attribution scores and a provided reference, or the variance of attribution scores across models to assess consistency. 
- `analyze_attr_scores.sh`: helper script for above
### Outputs 

#### 1. RMSE Calculation (if `--rmse` flag is set)
For each model in the ensemble, the script calculates the Root Mean Squared Error (RMSE) between the model's attribution scores and a reference attribution map. This provides a measure of similarity between each model's attribution patterns and the reference.

Each model’s RMSE values are saved as .npy files in the specified output directory.
```
<model_index>_top<top_n>_<enhancer>_<method>_euclidean_dist.npy
```
where `<model_index>` is the identifier for each model, `<top_n>` is the number of top predictions used, `<enhancer>` is the enhancer class specified (e.g., `Dev`), and `<method>` is the attribution method (either `saliency` or `shap`).

Each .npy file contains an array of RMSE values for each sequence analyzed, representing the Euclidean distance between the model’s attribution scores and the reference file across positions in each sequence.
#### 2. Variance Calculation (if `--var` flag is set)
When this option is enabled, the script calculates the variance of attribution scores across the ensemble of models. This measures consistency in attribution scores at each nucleotide and sequence position across models, followed by an aggregate measure across positions.

A single .npy file is generated for the variance calculations, saved in the specified output directory.
```
top<top_n>_<enhancer>_<method>_stdev.npy
```
The .npy file contains an array of standard deviations (computed from variances) across all analyzed positions, giving a summarized metric for sequence-level variance across the models. This output helps evaluate how much variation exists in attribution scores across the ensemble.

## 3. Visualizing results
- `DeepSTARR_attr_analysis.ipynb`: visualizes results of analyses performed in step #2. 
<!-- - `euclidean_distance_attribution_maps.ipynb`: plot results
- `plot_attribution_analysis_evoaug_deepstarr.ipynb`: visualize the results of the attribution analyses conducted for DeepSTARR ensembles & distilled models trained w/ EvoAug -->

# Generating shifted sequences 
Apply random mutagenesis, EvoAug, or random shuffling to STARR-seq test sequences to simulate a covariate shift. 

## Code 
- `get_ood_starrseq.ipynb`: Located in `data_preprocessing` directory. Generates shifted sequences and saves them as one-hot encoded sequences (shape (N,4)) to an HDF5 file for easy retrieval. 

# Training with dynamic augmentations
Train distilled models with dynamic augmentations. On each mini-batch, dynamically generate augmented sequences and either append or replace original training data. Target labels for augmented sequences are generated using an ensemble of models. Augmentation options: `random`, `mutagenesis`, `evoaug`

## Code
- `dynamic_aug.py`: Class definition for `DynamicAugModel()`
- `train_dynamic_aug_DeepSTARR.py`: Train distilled DeepSTARR models to activity and epistemic uncertainty predictions with dynamic augmentations. 
- `distill_DeepSTARR_dynamic_aug.sh`: Bash script that runs `train_dynamic_aug_DeepSTARR.py` to train distilled DeepSTARR models with dynamic augmentations.

## Outputs 
Located in output directory specified to `--out` argument of `train_dynamic_aug_DeepSTARR.py`. 

If `--append` flag is set: 
- `<ix>_DeepSTARR_<aug>_append_aug.h5`: Weights of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_append_historyDict_aug`: Training history of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_append_performance_aug.csv`: CSV file with performance metrics for distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_DeepSTARR_<aug>_append_finetune.h5`: Weights of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_append_historyDict_finetune`: Training history of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_append_performance_finetune.csv`: CSV file with performance metrics for distilled DeepSTARR model trained with specified dynamic augmentation. 
If `--append` flag is **not** set:
- `<ix>_DeepSTARR_<aug>_aug.h5`: Weights of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_historyDict_aug`: Training history of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_performance_aug.csv`: CSV file with performance metrics for distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_DeepSTARR_<aug>_finetune.h5`: Weights of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_historyDict_finetune`: Training history of distilled DeepSTARR model trained with specified dynamic augmentation. 
- `<ix>_<aug>_performance_finetune.csv`: CSV file with performance metrics for distilled DeepSTARR model trained with specified dynamic augmentation. 
<!-- # Interval coverage probability analysis
## Code 
- `prediction_interval_analysis.ipynb` -->

<!-- # Training models w/ evidential regression
This is done as part of the interval coverage probability analysis. An additional `--evidential` flag has been added to `train_DeepSTARR.py` and is run using `train_evidential_DeepSTARR.sh` -->

<!-- # Consistency analysis for attribution scores
`consistency_analysis_attribution_scores.ipynb` -->

# Evaluating different ensemble sizes

## 1. Train up to $N$ teacher models with standard training 
The following steps assume that the user has trained $N$ DeepSTARR models with standard training (generated with `train_DeepSTARR.py`), where $N$ is the largest ensemble size being evaluated for this analysis (in our case, $N=25$).

### Code 
See code for [Train ensemble of DeepSTARR models with standard training](#code).

### Outputs
See outputs of [Train ensemble of DeepSTARR models with standard training](#standard-training).
## 2. Evaluate performance of different ensemble sizes and get training data for distilled models 
These steps follow [Generate training data for distilled models](#2-generate-training-data-for-distilled-models) while also providing performance metrics for different ensemble sizes.

### Code 
- `ensemble_predict_DeepSTARR_over_ensemble_size.py`: Modified from `ensemble_predict_DeepSTARR.py` and `stdev_ensemble_predictions.py` to combine their functions and make predictions over max ensemble size and get ensemble average and standard deviation and performance metrics for various downsamples of max ensemble size (in our case, 2, 3, 4, 5, 10, 15, 20, 25). This script returns the ensemble average over the training sequences for all ensemble sizes evaluated as well as the ensemble standard deviation over all data splits and all ensemble sizes evaluated. 
- `eval_and_distill_DeepSTARR_over_ensemble_size.sh`: runs `ensemble_predict_DeepSTARR_over_ensemble_size.py` 
- `parse_DeepSTARR_distillation_data_over_ensemble_size.ipynb`: parse outputs of `ensemble_predict_DeepSTARR_over_ensemble_size.py` to generate HDF5 files for training distilled models; one HDF5 file per ensemble size evaluated. 
### Outputs
See outputs from [Step 2: Generate training data for distilled models](#2-generate-training-data-for-distilled-models).

## 3. Train distilled models based on different teacher ensemble sizes
Use the output of `parse_DeepSTARR_distillaion_data_over_ensemble_size.ipynb` to train distilled models based on different teacher ensemble sizes. 
### Code 
- `distill_DeepSTARR_with_std_over_ensemble_size.sh`: run `train_stdev_DeepSTARR.py` looping over different teacher ensemble sizes and the respective distillation training data associated with them
### Outputs
See outputs of [Step 3: Train distilled models with epistemic uncertainty estimates](#outputs-3).