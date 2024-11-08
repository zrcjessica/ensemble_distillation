# run DeepSTARR_ensemble_attr_analysis.py
# set DISTILLED to perform attribution analysis for distilled models (on activity output heads)
# set METHOD as saliency or shap to define method of attribution analysis
# set HEAD as mean/std/logvar (must be mean if distilled is false)
# set EPISTEMIC_METRIC as std/logvar (used if HEAD is not mean)

DISTILLED=true # analyzed distilled DeepSTARR Models 
METHOD=saliency # set saliency or shap
N_MODS=10
TOP_N=1000
ENHANCER=Hk # Dev/Hk
HEAD=mean # mean/std/logvar
EPISTEMIC_METRIC=std #std/logvar
MODELS_DIR=../results/DeepSTARR_ensemble_NEW
DATA=../data/DeepSTARR/Sequences_activity_all.h5
# define sequence data and path to models if analyzing distilled models 
if [ "$DISTILLED" = true ]; then 
	echo "analyzing distilled models"
	DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5
	MODELS_DIR=../results/DeepSTARR_ensemble_NEW/distilled_with_${EPISTEMIC_METRIC} # assume HEAD=std/logvar
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DISTILLED" = true ]; then
	# attr analysis for distilled models
	CUDA_VISIBLE_DEVICES=7 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --top_n $TOP_N --std --enhancer $ENHANCER --head $HEAD
else
	# typically only calculate average if looking at the ensemble, not the distilled models
	CUDA_VISIBLE_DEVICES=7 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average --top_n $TOP_N --enhancer $ENHANCER 
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top $TOP_N $ENHANCER $HEAD head on $MODELS_DIR ensemble (distilled, mean+std) completed successfully" &>/dev/null
		else
			slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (+ensemble avg of attr maps) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top $TOP_N $ENHANCER $HEAD head on $MODELS_DIR ensemble (distilled) exited with error code $exit_code"
		else
			slack "running $METHOD analysis for top $TOP_N $ENHANCER on $MODELS_DIR ensemble (+ensemble avg of attr maps) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"