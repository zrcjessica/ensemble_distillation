# run saliency analysis for top n Dev enhancers on an ensemble of DeepSTARR models
# set DISTILLED to perform attribution analysis for distilled models (w/ mean output)
# set METHOD as saliency or shap to define method of attribution analysis

DISTILLED=true # toggle flag
EPISTEMIC=true # true/false; determines whether model w/ epistemic prediction is evaluated
METHOD=saliency # set saliency or shap
N_MODS=10
TOP_N=1000
ENHANCER=Hk # Dev/Hk
HEAD=mean # mean/std/logvar; if STD is false then mean is default option
EPISTEMIC_METRIC=std #std/logvar
# MODELS_DIR=../results/DeepSTARR_lr-decay
MODELS_DIR=../results/DeepSTARR_ensemble_NEW
DATA=../data/DeepSTARR/Sequences_activity_all.h5
if [ "$DISTILLED" = true ]; then 
	echo "analyzing distilled models"
	if [ "$EPISTEMIC" = true ]; then
		DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5
		# MODELS_DIR=../results/DeepSTARR_lr-decay/distilled_with_${HEAD} # assume HEAD=std/logvar
		MODELS_DIR=../results/DeepSTARR_ensemble_NEW/distilled_with_${EPISTEMIC_METRIC} # assume HEAD=std/logvar
	else 
		# MODELS_DIR=../results/DeepSTARR_lr-decay/ensemble_distilled
		MODELS_DIR=../results/DeepSTARR_ensemble_NEW/ensemble_distilled
	fi
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DISTILLED" = true ]; then
	# attr analysis for distilled models
	if [ "$EPISTEMIC" = true ]; then
		# for distilled models w/ epistemic uncertainty
		CUDA_VISIBLE_DEVICES=7 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --top_n $TOP_N --std --enhancer $ENHANCER --head $HEAD
	else 
		# for distilled models w/ activity only; HEAD=mean
		CUDA_VISIBLE_DEVICES=7 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --top_n $TOP_N --enhancer $ENHANCER
	fi
else
	# typically only calculate average if looking at the ensemble, not the distilled models
	CUDA_VISIBLE_DEVICES=7 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average --top_n $TOP_N --enhancer $ENHANCER 
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			if [ "$STD" = true ]; then	
				slack "running $METHOD analysis for top $TOP_N $ENHANCER $HEAD head on $MODELS_DIR ensemble (distilled, mean+std) completed successfully" &>/dev/null
			else 
				slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (distilled) completed successfully" &>/dev/null
			fi
		else
			slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (+ensemble avg of attr maps) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			if [ "$STD" = true ]; then 
				slack "running $METHOD analysis for top $TOP_N $ENHANCER $HEAD head on $MODELS_DIR ensemble (distilled, mean+std) exited with error code $exit_code"
			else 
				slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (distilled) exited with error code $exit_code"
			fi
		else
			slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (+ensemble avg of attr maps) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"