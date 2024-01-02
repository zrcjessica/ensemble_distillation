# run saliency analysis for top 500 Dev enhancers on an ensemble of DeepSTARR models trained on DOWNSAMPLED training data (X_train)
# set DISTILLED to perform attribution analysis for distilled models
# set METHOD as saliency or shap to define method of attribution analysis

DISTILLED=true # toggle this flag 
METHOD=saliency # set as saliency or shap
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
N_MODS=10
PROJ_DIR=../results/DeepSTARR_lr-decay/
# if [ "$DISTILLED" = true ]; then PROJ_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled && echo "analyzing distilled models"; fi
# # MODELS_DIR=../results/DeepSTARR_lr-decay
# MODELS_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled
DATA=../data/DeepSTARR/Sequences_activity_all.h5


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# # CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --average
# CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD

if [ "$DISTILLED" = true ]; then
	echo "analyzing distilled downsampled models"
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		MODELS_DIR=${PROJ_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled
		CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD
	done
else
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		MODELS_DIR=${PROJ_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
		CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average 
	done
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (downsampled + distilled, trained with LR decay) completed successfully" &>/dev/null
		else
			slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (downsampled + no distillation, trained with LR decay) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (downsampled + distilled, trained with LR decay) exited with error code $exit_code"
		else
			slack "running saliency analysis for top 500 Dev on DeepSTARR_lr-decay ensemble (downsampled + no distillation, trained with LR decay) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"
