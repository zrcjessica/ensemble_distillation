# runs ensemble_predict_DeepSTARR.py

MODEL_DIR=../results/DeepSTARR_base
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA | simple_gpu_scheduler --gpus 4,5 