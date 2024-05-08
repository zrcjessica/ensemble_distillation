- [1. Train lentiMPRA models](#1-train-lentimpra-models)
  - [Scripts](#scripts)
- [2. Evaluate performance of ensemble](#2-evaluate-performance-of-ensemble)
  - [Scripts](#scripts-1)
- [3. Get ensemble average of training data](#3-get-ensemble-average-of-training-data)
  - [Scripts](#scripts-2)


# 1. Train lentiMPRA models
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