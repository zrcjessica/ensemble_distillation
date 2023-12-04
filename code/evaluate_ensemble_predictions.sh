# runs ensemble_predict_DeepSTARR.py

MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (full, trained with LR decay) completed successfully" &>/dev/null
	else
		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (full, trained with LR decay) exited with error code $exit_code"
	fi
fi
exit "$exit_code"