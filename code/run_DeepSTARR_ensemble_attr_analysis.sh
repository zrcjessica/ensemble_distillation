# run saliency analysis for top 500 Dev enhancers on an ensemble of DeepSTARR models
# set DISTILLED to perform attribution analysis for distilled models
# set METHOD as saliency or shap to define method of attribution analysis

DISTILLED=true # toggle flag
METHOD=saliency # set saliency or shap
N_MODS=10
MODELS_DIR=../results/DeepSTARR_lr-decay
if [ "$DISTILLED" = true ]; then MODELS_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled && echo "analyzing distilled models"; fi
# MODELS_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DISTILLED" = true ]; then
	CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD
else
	# typically only calculate average if looking at the ensemble, not the distilled models
	CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average
fi 

# # CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --average
# CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $method

# # message the user on slack if possible
# exit_code="$?"
# if command -v 'slack' &>/dev/null; then
#     if [ "$exit_code" -eq 0 ]; then
# 		slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) completed successfully" &>/dev/null
# 	else
# 		slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) exited with error code $exit_code"
# 	fi
# fi
# exit "$exit_code"

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) completed successfully" &>/dev/null
		else
			slack "running $METHOD analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (no distillation, trained with LR decay) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (distilled, trained with LR decay) exited with error code $exit_code"
		else
			slack "running $METHOD analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (no distillation, trained with LR decay) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"