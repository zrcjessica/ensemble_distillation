# run saliency analysis for top 500 Dev enhancers on an ensemble of DeepSTARR models

N_MODS=10
# MODELS_DIR=../results/DeepSTARR_lr-decay
MODELS_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_saliency_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --average
CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_saliency_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) completed successfully" &>/dev/null
	else
		slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) exited with error code $exit_code"
	fi
fi
exit "$exit_code"