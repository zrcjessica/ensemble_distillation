- [1. Train ResidualBind models (activity)](#1-train-residualbind-models-activity)
  - [Scripts](#scripts)
- [2. Evaluate performance of ensemble](#2-evaluate-performance-of-ensemble)
  - [Scripts](#scripts-1)
- [3. Get ensemble average of training data](#3-get-ensemble-average-of-training-data)
  - [Scripts](#scripts-2)
- [4. Train replicates of distilled models (mean)](#4-train-replicates-of-distilled-models-mean)
  - [Scripts](#scripts-3)
- [5. Train ensemble of ResidualBind models (mean+aleatoric std)](#5-train-ensemble-of-residualbind-models-meanaleatoric-std)
  - [Scripts](#scripts-4)
- [6. Distill replicates of ResidualBind models (activity+aleatoric uncertainty)](#6-distill-replicates-of-residualbind-models-activityaleatoric-uncertainty)
  - [Scripts](#scripts-5)
- [7. Distill replicates of ResidualBind models (mean+aleatoric+epistemic uncertainty)](#7-distill-replicates-of-residualbind-models-meanaleatoricepistemic-uncertainty)
  - [Scripts](#scripts-6)
- [Other](#other)
  - [Training with dynamic augmentations](#training-with-dynamic-augmentations)
    - [Scripts](#scripts-7)
  - [Training models w/ evidential regression](#training-models-w-evidential-regression)
  - [MPRAnn](#mprann)
  - [MPRAnn + heteroscedastic regression](#mprann--heteroscedastic-regression)


# 1. Train ResidualBind models (activity)
Train an ensemble of ResidualBind models on lentiMPRA data that predict activity.

## Scripts
- `train_lentiMPRA.py`: trains an ensemble of ResidualBind models; need to supply celltype (HepG2/K562)
- `train_lentiMPRA_ensemble.sh`: runs `train_lentiMPRA.py`; can toggle `evoaug` and `downsample` boolean variables to train with/without EvoAug or downsampling training data, respectively 

# 2. Evaluate performance of ensemble 
Evaluate performance of ensemble predictions on test set. Can be combined with [step 3](#3-get-ensemble-average-of-training-data) using the same script.

## Scripts
- `ensemble_predict_lentiMPRA.py`: use with `--eval` flag to evaluate ensemble average performance on test set for ResidualBind models
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

# 5. Train ensemble of ResidualBind models (mean+aleatoric std)
Train an ensemble of ResidualBind models on replicate mean and replicate std from lentiMPRA data. These models predict activity and aleatoric uncertainty (as std.). 

## Scripts 
- `parse_lentiMPRA_data_with_aleatoric.ipynb`: parse .h5 files for K562 and HepG2 with mean activity level and across-replicate std. of activity level 
- `train_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` with `--aleatoric` flag using .h5 files generated above 


# 6. Distill replicates of ResidualBind models (activity+aleatoric uncertainty)
Use average of ensemble average on training seqs to replace `y_train` and train distilled models. Average of aleatoric uncertainty predictions are used as new target aleatoric uncertainty values for training set seqs.

## Scripts
- `eval_and_distill_aleatoric_lentiMPRA.sh`: run `ensemble_predict_lentiMPRA.py` with `--aleatoric` flag to generate training data for distilled models
- `distill_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` to train distilled lentiMPRA models (mean+aleatoric std) using h5 file generated in [step 5](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std) and passing ensemble average .npy files generated above (`ensemble_avg_y_train.npy`) to `--distill` argument 

# 7. Distill replicates of ResidualBind models (mean+aleatoric+epistemic uncertainty)
Finally, train distilled models using ensemble trained in [step 5](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std) that predict epistemic uncertainty in addition to aleatoric uncertainty (as std). This requires getting the standard deviation of the ensemble's predictions for the activity level and using those as the epistemic uncertainty target values for the train, test, and validation sets. 

## Scripts
- `get_lentiMPRA_ensemble_std.sh`: run `get_lentiMPRA_ensemble_std.py` on ensemble of models w/ activity+aleatoric outputs
- `parse_lentiMPRA_data_with_epistemic.ipynb`: parse h5 file containing all data for distilling mean+aleatoric+epistemic models using ensemble mean for train seqs and ensemble std for train/test/val seqs 
  - ensemble std. was calculated for both activity and aleatoric outputs; last column is omitted when training distilled models, as epistemic output head describes epistemic uncertainty about activity prediction
- `train_distilled_lentiMPRA_with_epistemic.py`: replaces `train_lentiMPRA.py`, assumes `--aleatoric` and `--epistemic` are `True` and that h5 file provided to `--data` contains only data for distillation to bypass need for `--distill` flag and explicitly providing ensemble level metrics. 
- `distill_lentiMPRA_epistemic.sh`: runs above script

# Other

## Training with dynamic augmentations
Train distilled models with dynamic augmentations. On each mini-batch, dynamically generate augmented sequences and either append or replace original training data. Target labels for augmented sequences are generated using an ensemble of models. Augmentation options: `random`, `mutagenesis`, `evoaug`

### Scripts
- `dynamic_aug.py`: class definition for `DynamicAugModel()`
- `train_dynamic_aug_lentiMPRA.py`: train distilled lentiMPRA models with activity, aleatoric and epistemic uncertainty predictions using ensemble of DeepSTARR models with activity and aleatoric uncertainty outputs. 
- `distill_lentiMPRA_dynamic_aug.sh`: run `train_dynamic_aug_lentiMPRA.py`
- 
## Training models w/ evidential regression
This is done as part of the interval coverage probability analysis. An additional `--evidential` flag has been added to `train_lentiMPRA.py` and is run using `train_evidential_lentiMPRA.sh`

## MPRAnn
- `train_MPRAnn_ensemble.sh`: runs `train_MPRAnn.py` to train an ensemble of MPRAnn models; set `--aleatoric` flag to train with aleatoric uncertainty estimation
- `eval_and_distill_MPRAnn.sh`: runs `ensemble_predict_MPRAnn.py`
- `get_MPRAnn_ensemble_std.py`: calculate ensemble std on output heads, wrapper script is `get_MPRAnn_ensemble_std.sh`
- `parse_MPRAnn_distillation_data.ipynb`: generates .h5 file with updated train/test/val data for distilled models with activity + aleatoric + epistemic uncertainty outputs 
- `train_distilled_MPRAnn_with_epistemic.py`: train distilled MPRAnn models to predict activity + aleatoric + epistemic uncertainty using data from notebook above; wrapper script is `distill_MPRAnn_epistemic.sh`

## MPRAnn + heteroscedastic regression
- `train_heteroscedastic_MPRAnn.py` + `train_heteroscedastic_MPRAnn_ensemble.sh`: train ensemble of MPRAnn models with heteroscedastic regression on the lentiMPRA data that includes across replicate std. (makes it easier for evaluating performance of both activity and aleatoric uncertainty output heads )
- `ensemble_predict_heteroscedastic_MPRAnn.py` + `eval_and_distill_heteroscedastic_MPRAnn.sh`: get ensemble average on training data and performance of ensemble average on test set
- `get_heteroscedastic_MPRAnn_ensemble_std.sh` and `get_heteroscedastic_MPRAnn_ensemble_std.py`: calculate standard deviation of ensemble predictions for activity and aleatoric uncertainty 
