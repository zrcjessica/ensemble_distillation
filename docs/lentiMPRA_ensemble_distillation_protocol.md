- [1. Train ResidualBind models (activity)](#1-train-residualbind-models-activity)
  - [Scripts](#scripts)
  - [Outputs](#outputs)
    - [Trained lentiMPRA Models](#trained-lentimpra-models)
    - [Pickled Model Histories](#pickled-model-histories)
    - [Model Performance on Test Set](#model-performance-on-test-set)
    - [Additional Files](#additional-files)
- [2. Evaluate performance of ensemble](#2-evaluate-performance-of-ensemble)
  - [Scripts](#scripts-1)
  - [Outputs](#outputs-1)
- [3. Train ensemble of ResidualBind models (mean+aleatoric std)](#3-train-ensemble-of-residualbind-models-meanaleatoric-std)
  - [Scripts](#scripts-2)
  - [Outputs](#outputs-2)
    - [Data for training ensemble of ResidualBind models with aleatoric uncertainty prediction](#data-for-training-ensemble-of-residualbind-models-with-aleatoric-uncertainty-prediction)
    - [Ensemble of ResidualBind models with aleatoric uncertainty prediction](#ensemble-of-residualbind-models-with-aleatoric-uncertainty-prediction)
- [4. Get data for training distilled ResidualBind models](#4-get-data-for-training-distilled-residualbind-models)
  - [Code](#code)
  - [Outputs](#outputs-3)
    - [Ensemble averages](#ensemble-averages)
    - [Ensemble standard deviations](#ensemble-standard-deviations)
      - [Additional files for downsampled data](#additional-files-for-downsampled-data)
    - [Data for training distilled models](#data-for-training-distilled-models)
- [5. Distill replicates of ResidualBind models (mean+aleatoric+epistemic uncertainty)](#5-distill-replicates-of-residualbind-models-meanaleatoricepistemic-uncertainty)
  - [Scripts](#scripts-3)
  - [Outputs](#outputs-4)
- [Other analyses](#other-analyses)
  - [Training with dynamic augmentations](#training-with-dynamic-augmentations)
    - [Scripts](#scripts-4)
    - [Outputs](#outputs-5)
    - [If `--append` flag is set:](#if---append-flag-is-set)
    - [If `--append` flag is **not** set:](#if---append-flag-is-not-set)
    - [Additional Files](#additional-files-1)
  - [Training models w/ evidential regression](#training-models-w-evidential-regression)
    - [Scripts](#scripts-5)
    - [Outputs](#outputs-6)
  - [MPRAnn](#mprann)
    - [Scripts](#scripts-6)
    - [Outputs](#outputs-7)
  - [ResidualBind + heteroscedastic regression](#residualbind--heteroscedastic-regression)
    - [Scripts](#scripts-7)
    - [Outputs](#outputs-8)
  - [MPRAnn + heteroscedastic regression](#mprann--heteroscedastic-regression)


# 1. Train ResidualBind models (activity)
Train an ensemble of ResidualBind models with standard training to predict regulatory activity **only**. These models will provide baseline performance of individual ResidualBind models and an ensemble of ResidualBind models. 

## Scripts
- `train_lentiMPRA.py`: Trains ResidualBind models on lentiMPRA data for specified cell type (HepG2/K562) and evaluate their performance. 
- `train_lentiMPRA_ensemble.sh`: runs `train_lentiMPRA.py`; can toggle `evoaug` and `downsample` boolean variables to train with/without EvoAug or downsampling training data, respectively. 

## Outputs
### Trained lentiMPRA Models
- `<model_ix>_lentiMPRA.h5`: Trained lentiMPRA model without augmentations (saved with `model.save()`).
- `<model_ix>_lentiMPRA_aug_weights.h5`: Trained lentiMPRA model weights with augmentations.
- `<model_ix>_lentiMPRA_finetune.h5`: Fine-tuned model weights without augmentations.

### Pickled Model Histories
- `<model_ix>_historyDict`: Training history for models trained without augmentations.
- `<model_ix>_historyDict_aug`: Training history for models trained with augmentations.
- `<model_ix>_historyDict_finetune`: Training history for fine-tuned models.

### Model Performance on Test Set
- `<model_ix>_performance.csv`: Performance of models trained without augmentations.
- `<model_ix>_performance_aug.csv`: Performance of models trained with augmentations.
- `<model_ix>_performance_finetune.csv`: Performance of fine-tuned models.

### Additional Files
- `config.yaml`: Updated configuration file with all training parameters (not specific to a particular model index).

# 2. Evaluate performance of ensemble 
Evaluate performance of ensemble average predictions on held out test sequences. This provides benchmark ensemble performance. 
<!-- Can be combined with [step 3](#3-get-ensemble-average-of-training-data) using the same script. -->

## Scripts
- `ensemble_predict_lentiMPRA.py`: Use with `--eval` flag to evaluate ensemble average performance on test set for ResidualBind models
- `eval_and_distill_ensemble_lentiMPRA.sh`: Runs `ensemble_predict_lentiMPRA.py` with `--eval` and `--distill` flags set. Only output of `--eval` flag is needed to evaluate ensemble performance. 
## Outputs
- `ensemble_avg_y_train.npy`: Ensemble average predictions on training sequences (`X_train`), generated if the `--distill` flag is set.
- `ensemble_performance_avg.csv`: Ensemble average performance on test sequences (`X_test`), saved if the `--eval` flag is set.


<!-- # 3. Get ensemble average of training data
Calculate ensemble average performance on training sequences and save to .npy file. The output is used to train the distilled models. Can be combined with [step 2](#2-evaluate-performance-of-ensemble) using the same script. 

## Scripts
- `ensemble_predict_lentiMPRA.py`: use with `--distill` flag to calculate average of ensemble predictions on training data and save it to an .npy file that can be provided to `train_lentiMPRA.py` to train distilled models 
- `eval_and_distill_ensemble_lentiMPRA.sh`: runs `ensemble_predict_lentiMPRA.py` with `--eval` and `--distill` flags set -->

<!-- # 4. Train replicates of distilled models (mean)
For HepG2 and K562.

## Scripts 
- `train_lentiMPRA_distilled.sh`: runs `train_lentiMPRA.py` with `ensemble_avg_y_train.npy` file generated in [step 3](#3-get-ensemble-average-of-training-data) as input to the `--distill` flag
  - set `downsample` equal to `true`/`false` to train distilled models on subsets of the trainig data or not 
  - set `evoaug` equal to `true`/`false` to train distilled models w/ EvoAug -->

# 3. Train ensemble of ResidualBind models (mean+aleatoric std)
Train an ensemble of ResidualBind models on the mean and standard deviation over biological replicates in the lentiMPRA data. These models predict activity and aleatoric uncertainty, and will be distilled to produce student models that can predict both aleatoric and epistemic uncertainty. 

## Scripts 
- `parse_lentiMPRA_data_with_aleatoric.ipynb`: Parse HDF5 files for K562 and HepG2 with mean and standard deviation of experimental activity values across biological replicates in lentiMPRA data (target labels have shape (N,2)).
- `train_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` with `--aleatoric` flag using HDF5 files generated above as input to `--data`.
## Outputs 
### Data for training ensemble of ResidualBind models with aleatoric uncertainty prediction
The Jupyter notebook generates an HDF5 file that contains lentiMPRA data in train/test/val splits, where target labels have shape (N,2) and second column represents replicate standard deviation (aleatoric uncertainty).
### Ensemble of ResidualBind models with aleatoric uncertainty prediction
See outputs of [1. Train ResidualBind models (activity)](#outputs) for outputs of script for training ensemble of ResidualBind models with activity and aleatoric uncertainty outputs.

# 4. Get data for training distilled ResidualBind models  
For the ensemble of ResidualBind models that predict activity and aleatoric uncertainty, we need the ensemble average of its predictions across training sequence as well as the ensemble standard deviation of its predictions across all splits to generate a new HDF5 file that contains data for training and evaluating the distilled models, which will predict activity, aleatoric, and epistemic uncertainty. 

## Code 
- `eval_and_distill_aleatoric_lentiMPRA.sh`: Run `ensemble_predict_lentiMPRA.py` with `--aleatoric` flag to calculate ensemble average for both activity and aleatoric uncertainty output heads. Uses `--distill` and `--eval` flags, so ensemble average performance on test sequences is evaluated as well. 
- `get_lentiMPRA_ensemble_std.py`: Calculate ensemble standard deviation on train/test/val splits of lentiMPRA data for specified cell type and saves output to .npy files, which will be used to generate HDF5 file for training distilled models. 
- `get_lentiMPRA_ensemble_std.sh`: run `get_lentiMPRA_ensemble_std.py` on ensemble of models w/ activity+aleatoric outputs
- `parse_lentiMPRA_data_with_epistemic.ipynb`: Parse HDF5 file containing all data for distilling mean+aleatoric+epistemic models using ensemble mean for train seqs and ensemble standard deviation for train/test/val seqs. HDF5 files are specific to downsampling proportion. 
  - Ensemble standard deviation was calculated for both activity and aleatoric output heads of teacher ensemble. We will omit the latter quantity when training the distilled models, as the epistemic output head describes the epistemic uncertainty for regulatory activity.
## Outputs
### Ensemble averages
See outputs of [2. Evaluate performance of ensemble](#outputs-1) for outputs of `ensemble_predict_lentiMPRA.py`. 

### Ensemble standard deviations
- `<celltype>_ensemble_std_train.npy`: Ensemble standard deviation of predictions on training sequences (`X_train`).
- `<celltype>_ensemble_std_test.npy`: Ensemble standard deviation of predictions on test sequences (`X_test`).
- `<celltype>_ensemble_std_val.npy`: Ensemble standard deviation of predictions on validation sequences (`X_val`).
#### Additional files for downsampled data
If the `--downsample` flag is set:
- `<celltype>_downsample<downsample_value>_ensemble_std_train.npy`: Ensemble standard deviation of predictions on downsampled training sequences.
- `<celltype>_downsample<downsample_value>_ensemble_std_test.npy`: Ensemble standard deviation of predictions on downsampled test sequences.
- `<celltype>_downsample<downsample_value>_ensemble_std_val.npy`: Ensemble standard deviation of predictions on downsampled validation sequences.
### Data for training distilled models
Jupyter notebook produces an HDF5 file containing train/test/val splits for training distilled models. Target labels should have shape (N,4) where the 3rd and 4th columns store the ensemble standard deviations of the activity and aleatoric uncertainty output heads, respectively. In practice, the fourth column is not used for training the distilled models. The first and second columns of the target labels for the training split represent the ensemble averages for the activity and aleatoric uncertainty output heads, respectively.

<!-- # 6. Distill replicates of ResidualBind models (activity+aleatoric uncertainty)
Use average of ensemble average on training seqs to replace `y_train` and train distilled models. Average of aleatoric uncertainty predictions are used as new target aleatoric uncertainty values for training set seqs.

## Scripts
- `eval_and_distill_aleatoric_lentiMPRA.sh`: run `ensemble_predict_lentiMPRA.py` with `--aleatoric` flag to generate training data for distilled models
- `distill_lentiMPRA_aleatoric.sh`: run `train_lentiMPRA.py` to train distilled lentiMPRA models (mean+aleatoric std) using h5 file generated in [step 5](#5-train-ensemble-of-lentimpra-models-meanaleatoric-std) and passing ensemble average .npy files generated above (`ensemble_avg_y_train.npy`) to `--distill` argument  -->

# 5. Distill replicates of ResidualBind models (mean+aleatoric+epistemic uncertainty)
Finally, train distilled models using ensemble of models trained in [step 3](#3-train-ensemble-of-residualbind-models-meanaleatoric-std). These distilled models will predict epistemic uncertainty in addition to aleatoric uncertainty. These models are trained on the HDF5 produced in [step 4](#4-get-data-for-training-distilled-residualbind-models).

## Scripts
- `train_distilled_lentiMPRA_with_epistemic.py`: Replaces `train_lentiMPRA.py`, assumes `--aleatoric` and `--epistemic` are `True` and that HDF5 file provided to `--data` contains only data for distillation, bypassing the need for the `--distill` flag and explicitly providing ensemble level metrics. 
- `distill_lentiMPRA_epistemic.sh`: Runs `train_distilled_lentiMPRA_with_epistemic.py` to train distilled models.
## Outputs 
- `<model_ix>_ResidualBind_aug_weights.h5`: Model weights trained with augmentations (EvoAug).
- `<model_ix>_ResidualBind_finetune.h5`: Fine-tuned model weights after training without augmentations.
- `<model_ix>_historyDict_aug`: Pickled training history for models trained with augmentations.
- `<model_ix>_historyDict_finetune`: Pickled training history for fine-tuned models.
- `<model_ix>_performance_aug.csv`: Performance of models trained with augmentations on test sequences.
- `<model_ix>_performance_finetune.csv`: Performance of fine-tuned models on test sequences.
- `config.yaml`: Updated configuration file with training parameters.

# Other analyses

## Training with dynamic augmentations
Train distilled models with dynamic augmentations. On each mini-batch, dynamically generate augmented sequences and either append or replace original training data. Target labels for augmented sequences are generated using an ensemble of models. Augmentation options: `random`, `mutagenesis`, `evoaug`

### Scripts
- `dynamic_aug.py`: class definition for `DynamicAugModel()`
- `train_dynamic_aug_lentiMPRA.py`: train distilled lentiMPRA models with activity, aleatoric and epistemic uncertainty predictions using ensemble of DeepSTARR models with activity and aleatoric uncertainty outputs. 
- `distill_lentiMPRA_dynamic_aug.sh`: run `train_dynamic_aug_lentiMPRA.py`

### Outputs
Located in the output directory specified by the `--out` argument.

### If `--append` flag is set:
- `<ix>_lentiMPRA_<aug>_append_aug.h5`: Weights of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_<aug>_append_historyDict_aug`: Pickled training history of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_<aug>_append_performance_aug.csv`: CSV file containing performance metrics of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_lentiMPRA_<aug>_append_finetune.h5`: Weights of the distilled ResidualBind model fine-tuned after training with the specified dynamic augmentation.
- `<ix>_<aug>_append_historyDict_finetune`: Pickled training history of the fine-tuned ResidualBind model.
- `<ix>_<aug>_append_performance_finetune.csv`: CSV file containing performance metrics of the fine-tuned ResidualBind model.

### If `--append` flag is **not** set:
- `<ix>_lentiMPRA_<aug>_aug.h5`: Weights of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_<aug>_historyDict_aug`: Pickled training history of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_<aug>_performance_aug.csv`: CSV file containing performance metrics of the distilled ResidualBind model trained with the specified dynamic augmentation.
- `<ix>_lentiMPRA_<aug>_finetune.h5`: Weights of the distilled ResidualBind model fine-tuned after training with the specified dynamic augmentation.
- `<ix>_<aug>_historyDict_finetune`: Pickled training history of the fine-tuned ResidualBind model.
- `<ix>_<aug>_performance_finetune.csv`: CSV file containing performance metrics of the fine-tuned ResidualBind model.

### Additional Files
- `config.yaml`: Updated configuration file with training parameters.

## Training models w/ evidential regression
These models are used for the interval coverage probability analysis. 
### Scripts
- `train_lentiMPRA.py`: Set `--evidential` flag to train with evidential regression. These models will have an additional output head that predicts aleatoric uncertainty as log variance. 
- `train_evidential_lentiMPRA.sh`: Runs `train_lentiMPRA.py` to train ResidualBind models with evidential regression. 
### Outputs
See outputs of [1. Train ResidualBind models (activity)](#outputs).

## MPRAnn
### Scripts
- `train_MPRAnn_ensemble.sh`: runs `train_MPRAnn.py` to train an ensemble of MPRAnn models; set `--aleatoric` flag to train with aleatoric uncertainty estimation
- `eval_and_distill_MPRAnn.sh`: runs `ensemble_predict_MPRAnn.py`
- `get_MPRAnn_ensemble_std.py`: calculate ensemble std on output heads, wrapper script is `get_MPRAnn_ensemble_std.sh`
- `parse_MPRAnn_distillation_data.ipynb`: generates HDF5 file with updated train/test/val data for distilled models with activity + aleatoric + epistemic uncertainty outputs 
- `train_distilled_MPRAnn_with_epistemic.py`: Train distilled MPRAnn models to predict activity + aleatoric + epistemic uncertainty using data from notebook above
- `distill_MPRAnn_epistemic.sh`: Runs `train_distilled_MPRAnn_with_epistemic.py` to train distilled MPRAnn models. 
### Outputs
- `<ix>_MPRAnn.h5`: Trained MPRAnn model (saved with `model.save()`), if evidential loss is not used.
- `<ix>_historyDict`: Training history for the MPRAnn model without augmentations.
- `<ix>_performance.csv`: Performance of MPRAnn models on the test sequences without augmentations.
- `config.yaml`: Updated configuration file with training parameters.

## ResidualBind + heteroscedastic regression
### Scripts 
- `train_heteroscedastic_ResidualBind_ensemble.sh`: runs `train_heteroscedastic_ResidualBind.py` to train an ensemble of ResidualBind models with heteroscedastic regression; these models predict activity and aleatoric uncertainty and are trained on the data that contains both activity and technical variation, but doesn't use the technical variation measures for calculating losses.
- `eval_and_distill_heteroscedastic_ResidualBind.sh`: runs `ensemble_predict_heteroscedastic_ResidualBind.py` to 1) get average of ensemble's predictions on training set, and 2) evaluate performance of average of ensemble's predictions on test set 
- `get_heteroscedastic_ResidualBind_ensemble_std.sh` and `get_heteroscedastic_ResidualBind_ensemble_std.py`: calculate standard deviation of ensemble predictions for activity and aleatoric uncertainty  
- `parse_lentiMPRA_data_with_epistemic.ipynb`: same as notebook used to parse data for distilling ResidualBind; set `heteroscedastic = True` and `evoaug = False`
- `distill_heteroscedastic_ResidualBind.sh`: Runs `train_distilled_lentiMPRA_with_epistemic.py` to train distilled ResidualBind models based on teacher ensemble of ResidualBind models trained with heteroscedastic regression. 
### Outputs 

## MPRAnn + heteroscedastic regression
- `train_heteroscedastic_MPRAnn.py` + `train_heteroscedastic_MPRAnn_ensemble.sh`: train ensemble of MPRAnn models with heteroscedastic regression on the lentiMPRA data that includes across replicate std. (makes it easier for evaluating performance of both activity and aleatoric uncertainty output heads )
- `ensemble_predict_heteroscedastic_MPRAnn.py` + `eval_and_distill_heteroscedastic_MPRAnn.sh`: get ensemble average on training data and performance of ensemble average on test set
- `get_heteroscedastic_MPRAnn_ensemble_std.sh` and `get_heteroscedastic_MPRAnn_ensemble_std.py`: calculate standard deviation of ensemble predictions for activity and aleatoric uncertainty 
- `parse_MPRAnn_distillation_data.ipynb`: generates HDF5 file with updated train/test/val data for distilled models with activity + aleatoric + epistemic uncertainty outputs 
- `train_distilled_MPRAnn_with_epistemic.py`: train distilled MPRAnn models to predict activity + aleatoric + epistemic uncertainty using data from notebook above; wrapper script is `distill_MPRAnn_epistemic.sh` with `heteroscedastic=true` set; specify training data distilled from models trained with heteroscedastic regression