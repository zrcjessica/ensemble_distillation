- [1. Train lentiMPRA models (mean)](#1-train-lentimpra-models-mean)
  - [Scripts](#scripts)
- [2. Evaluate performance of ensemble](#2-evaluate-performance-of-ensemble)
  - [Scripts](#scripts-1)
- [3. Get ensemble average of training data](#3-get-ensemble-average-of-training-data)
  - [Scripts](#scripts-2)
- [4. Train replicates of distilled models (mean)](#4-train-replicates-of-distilled-models-mean)
  - [Scripts](#scripts-3)
- [5. Train ensemble of lentiMPRA models (mean+aleatoric std)](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std)
  - [Scripts](#scripts-4)
- [6. Distill replicates of lentiMPRA models (mean+aleatoric std)](#6-distill-replicates-of-lentimpra-models-meanaleatoric-std)
  - [Scripts](#scripts-5)
- [7. Distill replicates of lentiMPRA models (mean+aleatoric+epistemic)](#7-distill-replicates-of-lentimpra-models-meanaleatoricepistemic)
  - [Scripts](#scripts-6)


# 1. Train lentiMPRA models (mean)
Train an ensemble of lentiMPRA models. 

## Scripts
- `train_lentiMPRA.py`: trains an ensemble of lentiMPRA models; need to supply celltype (HepG2/K562)
- `train_lentiMPRA_ensemble.sh`: runs `train_lentiMPRA.py`; can toggle `evoaug` and `downsample` boolean variables to train with/without EvoAug or downsampling training data, respectively 

# 2. Evaluate performance of ensemble 
Evaluate performance of ensemble predictions on test set. Can be combined with [step 3](#3-get-ensemble-average-of-training-data) using the same script.

## Scripts
- `ensemble_predict_lentiMPRA.py`: use with `--eval` flag to evaluate ensemble average performance on test set for lentiMPRA models
- `eval_and_distill_ensemble_lentiMPRA.sh`: runs `ensemble_predict_lentiMPRA.py` with `--eval` and `--distill` flags set 

# 3. Get ensemble average of training data
The output is used to train the distilled models. Can be combined with [step 2](#2-evaluate-performance-of-ensemble) using the same script. 

## Scripts
- `ensemble_predict_lentiMPRA.py`: use with `--distill` flag to calculate average of ensemble predictions on training data and save it to an .npy file that can be provided to `train_lentiMPRA.py` to train distilled models 
- `eval_and_distill_ensemble_lentiMPRA.sh`: runs `ensemble_predict_lentiMPRA.py` with `--eval` and `--distill` flags set

# 4. Train replicates of distilled models (mean)
For HepG2 and K562.

## Scripts 
- `train_lentiMPRA_distilled.sh`: runs `train_lentiMPRA.py` with `ensemble_avg_y_train.npy` file generated in [step 3](#3-get-ensemble-average-of-training-data) as input to the `--distill` flag
  - set `downsample` equal to `true`/`false` to train distilled models on subsets of the trainig data or not 
  - set `evoaug` equal to `true`/`false` to train distilled models w/ EvoAug

# 5. Train ensemble of lentiMPRA models (mean+aleatoric std)
Train an ensemble of lentiMPRA models with mean + aleatoric std. information. 

## Scripts 
- `parse_lentiMPRA_data_with_std.ipynb`: parse .h5 files for K562 and HepG2 with mean activity level and across-replicate std. of activity level 
- `train_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` with `--aleatoric` flag using .h5 files generated above 


# 6. Distill replicates of lentiMPRA models (mean+aleatoric std)
Use average of ensemble average on training seqs to replace `y_train` and train distilled models. Average of aleatoric uncertainty predictions are used as new target aleatoric uncertainty values for training set seqs.

## Scripts
- `eval_and_distill_aleatoric_lentiMPRA.sh`: run `ensemble_predict_lentiMPRA.py` with `--aleatoric` flag to generate training data for distilled models
- `distill_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` to train distilled lentiMPRA models (mean+aleatoric std) using h5 file generated in [step 5](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std) and passing ensemble average .npy files generated above (`ensemble_avg_y_train.npy`) to `--distill` argument 

# 7. Distill replicates of lentiMPRA models (mean+aleatoric+epistemic)
Finally, train distilled models using ensemble trained in [step 5](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std) that predict epistemic uncertainty in addition to aleatoric uncertainty. This requires getting the standard deviation of the ensemble's predictions for the activity level and using those as the epistemic uncertainty target values for the train, test, and validation sets. 

## Scripts
- `get_lentiMPRA_ensemble_std.sh`: run `get_lentiMPRA_ensemble_std.py` on ensemble of models w/ activity+aleatoric outputs
- `parse_lentiMPRA_data_with_epistemic.ipynb`: parse h5 file containing all data for distilling mean+aleatoric+epistemic models using ensemble mean for train seqs and ensemble std for train/test/val seqs 
  - ensemble std. was calculated for both activity and aleatoric outputs; last column is omitted when training distilled models, as epistemic output head describes epistemic uncertainty about activity prediction
- `train_distilled_lentiMPRA_with_epistemic.py`: replaces `train_lentiMPRA.py`, assumes `--aleatoric` and `--epistemic` are `True` and that h5 file provided to `--data` contains only data for distillation to bypass need for `--distill` flag and explicitly providing ensemble level metrics. 
- `distill_lentiMPRA_epistemic.sh`: runs above script