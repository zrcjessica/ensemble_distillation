# DeepSTARR ensemble distillation workflow and scripts
- [DeepSTARR ensemble distillation workflow and scripts](#deepstarr-ensemble-distillation-workflow-and-scripts)
- [1. Train replicates of original DeepSTARR model](#1-train-replicates-of-original-deepstarr-model)
  - [Scripts](#scripts)
  - [Outputs](#outputs)
    - [Trained w/ EvoAug](#trained-w-evoaug)
- [2. Generate training data for distilled models](#2-generate-training-data-for-distilled-models)
  - [Scripts](#scripts-1)
  - [Outputs](#outputs-1)
- [3. Evaluate model performance](#3-evaluate-model-performance)
  - [Scripts](#scripts-2)
  - [Outputs](#outputs-2)
- [4. Train distilled models](#4-train-distilled-models)
  - [Scripts](#scripts-3)
  - [Outputs](#outputs-3)
- [5. Evaluate performance of distilled models on test set](#5-evaluate-performance-of-distilled-models-on-test-set)
  - [Scripts](#scripts-4)
- [6. Attribution analysis](#6-attribution-analysis)
  - [Scripts](#scripts-5)
- [7. Analyze attribution scores](#7-analyze-attribution-scores)
  - [Scripts](#scripts-6)
  - [Plotting](#plotting)
- [8. Get ensemble standard deviation](#8-get-ensemble-standard-deviation)
  - [Scripts](#scripts-7)
- [9. Make an h5 file that contains the DeepSTARR training data and the ensemble standard deviation](#9-make-an-h5-file-that-contains-the-deepstarr-training-data-and-the-ensemble-standard-deviation)
  - [Scripts](#scripts-8)
  - [Outputs](#outputs-4)
- [10. Train distilled model with stdev. prediction](#10-train-distilled-model-with-stdev-prediction)
  - [Scripts](#scripts-9)

Note: all bash scripts use a Slack bot to message user upon completion. To use this feature, refer to the Slack docs: [Sending messages using incoming webhooks](https://api.slack.com/messaging/webhooks)

# 1. Train replicates of original DeepSTARR model
- on entire training dataset
- on subsets of training dataset

## Scripts
- `train_DeepSTARR.py`: train an ensemble of DeepSTARR models
- `train_DeepSTARR_ensemble.sh`: runs `train_DeepSTARR.py`; for training on entire training dataset. Toggle `evoaug` variable to train with/without EvoAug.
- `train_DeepSTARR_ensemble_downsampled.sh`: runs `train_DeepSTARR.py` with `--downsample` flag set to various values; for training on subsets of the training dataset. Toggle `evoaug` variable to train with/without EvoAug
  
## Outputs
- `[model_ix]_DeepSTARR.h5`: model saved to h5 file w/ `model.save()`
- `[model_ix]_historyDict`: pickled model history (`history.history`)
- `[model_ix]_performance.csv`: model performance on test set 
- `[model_ix]_loss_curves.png` if run with `--plot` flag

Outputs for dow`nsampled models are saved in a subdirectory named according to downsampling proportion (as defined in bash script).

### Trained w/ EvoAug
- `[model_ix]_DeepSTARR_aug_weights.h5`: weights of model trained w/ EvoAug (saved w/ `model.save_weights()`)
- `[model_ix]_historyDict_aug`: pickled weights of augmented model 
- `[model_ix]_DeepSTARR_finetune.h5`: weights of finetuned model (saved w/ `model.save_weights()`)
- `[model_ix]_historyDict_finetune`: pickled history of finetuned model 
- `[model_ix]_performance_finetune.csv`: performance of finetuned model on test set
  
# 2. Generate training data for distilled models 
Make predictions with the replicate models on the training dataset. If using a model trained on a subset of the training dataset, make predictions on the same examples that the model was trained on. 

## Scripts
- `ensemble_predict_DeepSTARR.py` with `--distill` flag set 
- `get_train_data_distilled.sh`
  - toggle `DOWNSAMPLED` true/false for models trained on entirety vs. subsets of training data 
- `eval_and_distill_ensemble.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set (prototype script, not used yet)
- `run_ensemble_predict_DeepSTARR_evoaug.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set for DeepSTARR ensembles trained w/ EvoAug
  
## Outputs
- `ensemble_avg_y_train.npy`: ensemble average predictions on train seqs

# 3. Evaluate model performance 
- for original models
  - trained on all training data
  - trained on subsets of training data
- for ensemble average
  - for ensembles trained on all training data
  - for ensembles trained on subsetes of training data
- for distilled models 
  - trained on all training data
  - trained on subsets of training data
## Scripts
- `ensemble_predict_DeepSTARR.py` with `--eval` flag 
- `evaluate_ensemble_predictions.sh`
- `eval_and_distill_ensemble.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set (prototype script, not used yet)
- `run_ensemble_predict_DeepSTARR_evoaug.sh`: runs `ensemble_predict_DeepSTARR.py` with both `--eval` and `--distill` flags set for DeepSTARR ensembles trained w/ EvoAug
## Outputs
- `ensemble_performance_avg.csv`
- `[model_ix]_pred_scatterplot.png`: scatterplot of predicted vs. true values for test set
- `avg_pred_scatterplot.png` if `--plot` flag set

# 4. Train distilled models 
- trained on all training data
- trained on subsets of training data, specifically using the same samples in the subsets used to train the downsampled ensembles... the random seed set in line 15 of `utils.py` *should* account for this. 

Requires output of step 2 (`ensemble_avg_y_train.npy`).

## Scripts
- `train_DeepSTARR.py`: with `--distill` flag set to `ensemble_avg_y_train.py` corresponding to training data size (full or downsampled)
- `train_DeepSTARR_distilled.sh`: for training distilled models on entire training dataset
- `train_DeepSTARR_distilled_downsampled.sh`: above, for downsampled training data
- `train_DeepSTARR_distilled_downsample_one_case.sh`: somewhat temporary script created to train downsampled models at p=0.75 because for some reason it was skipped
  
## Outputs
Same as [outputs of `train_DeepSTARR.py`](#outputs) but should be saved in a directory corresponding to ensemble distilled models. 

# 5. Evaluate performance of distilled models on test set 
- for distilled models trained on full training dataset
- for downsampled distilled models trained on subset of training data

## Scripts
- `ensemble_predict_DeepSTARR.py` with `--eval` flag (from [step 3](#scripts-2))
- `eval_ensemble_preds_distilled.sh`: bash script specific for distilled models

# 6. Attribution analysis
With saliency and SHAP. 

## Scripts
- `DeepSTARR_ensemble_attr_analysis.py`
- `run_DeepSTARR_ensemble_attr_analysis.sh`: runs attribution analysis on DeepSTARR models trained with full dataset
  - `run_DeepSTARR_ensemble_attr_analsysis_downsampled.sh`: runs attribution analysis on DeepSTARR models trained w/ downsampled training dataset
  - `run_evoaug_DeepSTARR_ensemble_attr_analysis.sh`: runs attribution analysis on DeepSTARR models trained w/ evoaug on full dataset; has been modified to include boolean flags for downsampled and distilled models

# 7. Analyze attribution scores
1. RMSE from ensemble average attribution score
2. Standard deviation of attribution scores
## Scripts
- `analyze_attr_scores.py`: If `--rmse` flag set, calculates RMSE b/w attribution scores of individual models and a reference set of attribution scores (i.e. ensemble average, needs to be previously calculated, saved, and provided - this can be done in step 5 if `--average` flag set). If `--var` flag set, calculates variance in attribution scores across models. Default activity: performs analysis for attribution scores of top 500 seqs with the largest ground truth Dev enhancer activity
- `analyze_attr_scores.sh`: helper script for above
  
## Plotting 
- `euclidean_distance_attribution_maps.ipynb`: plot results
- `plot_attribution_analysis_evoaug_deepstarr.ipynb`: visualize the results of the attribution analyses conducted for DeepSTARR ensembles & distilled models trained w/ EvoAug
- `variance_attribution_maps.ipynb`: (unclean) used to visualize results of attribution analyses conducted for DeepSTARR ensembles & distilled models (no EvoAug)

# 8. Get ensemble standard deviation 
Get the ensemble standard deviation for all data (train/test/val) to train & evaluate distilled models with uncertainty estimation. 

## Scripts
- `stdev_ensemble_predictions.py`: make predictions on train/test/val data with DeepSTARR ensemble and save ensemble standard deviation to file
- `generate_stdev_dataset.sh`: run the above script

# 9. Make an h5 file that contains the DeepSTARR training data and the ensemble standard deviation 
This will be used to train distilled models with uncertainty estimation.

## Scripts
- `parse_DeepSTARR_data.ipynb`: (in `data_preprocessing` dir)

## Outputs
Currently using file `all_data_with_ensemble_metrics_hierarchical.h5`.

# 10. Train distilled model with stdev. prediction

## Scripts
- `train_stdev_DeepSTARR.py`: python script for training distilled model w/ stdev. prediction (modified from `train_DeepSTARR.py`)
- `distill_DeepSTARR_with_std.sh`: run `train_stdev_DeepSTARR.py` 
  - can toggle `evoaug` and `downsample` boolean variables accordingly
- `distill_evoaug_DeepSTARR_with_std.sh`: run `train_stdev_DeepSTARR.py` with `--evoaug` flag (has now been wrapped into above script)
- `distil_DeepSTARR_with_logvar.sh`: runs `train_stdev_DeepSTARR.py` w/ `--logvar` flag (currently exploratory, in .gitignore)
  