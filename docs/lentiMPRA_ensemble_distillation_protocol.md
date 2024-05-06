# 1. Train lentiMPRA models
Train an ensemble of lentiMPRA models. 

## Scripts
- `train_lentiMPRA.py`: trains an ensemble of lentiMPRA models; need to supply celltype (HepG2/K562)
- `train_lentiMPRA_ensemble.sh`: runs `train_lentiMPRA.py`; can toggle `evoaug` and `downsample` boolean variables to train with/without EvoAug or downsampling training data, respectively 
