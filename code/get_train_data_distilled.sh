# runs get_train_data_for_distillation.py
# for DeepSTARR ensemble trained on full training data

MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

CUDA_VISIBLE_DEVICES=4,5 python get_train_data_for_distillation.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "running get_train_data_for_distillation.py for DeepSTARR_lr-decay (no downsampling) completed successfully" &>/dev/null
	else
		slack "running get_train_data_for_distillation.py for DeepSTARR_lr-decay (no downsampling) exited with error code $exit_code"
	fi
fi
exit "$exit_code"