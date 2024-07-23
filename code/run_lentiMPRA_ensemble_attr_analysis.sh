# run saliency analysis for top n seqs on an ensemble of lentiMPRA models

DISTILLED=true # toggle flag
METHOD=saliency # set saliency or shap
N_MODS=10
TOP_N=1000
CELLTYPE=HepG2
HEAD=mean # mean/std/logvar; if STD is false then mean is default option
MODELS_DIR=../results/lentiMPRA/${CELLTYPE}
DATA=../data/lentiMPRA/${CELLTYPE}_data_with_aleatoric.h5
if [ "$DISTILLED" = true ]; then 
	echo "analyzing distilled models"
	MODELS_DIR=../results/distilled_lentiMPRA_epistemic/${CELLTYPE} 
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DISTILLED" = true ]; then
	CUDA_VISIBLE_DEVICES=2 python lentiMPRA_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --top_n $TOP_N --celltype $CELLTYPE --head $HEAD 
else
	# typically only calculate average if looking at the ensemble, not the distilled models
	CUDA_VISIBLE_DEVICES=2 python lentiMPRA_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average --top_n $TOP_N --celltype $CELLTYPE 
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top $TOP_N $CELLTYPE $HEAD head on $MODELS_DIR ensemble (distilled, mean+aleatoric+epistemic) completed successfully" &>/dev/null
		else
			slack "running $METHOD analysis for top $TOP_N $CELLTYPE on $MODELS_DIR ensemble (+ensemble avg of attr maps) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top $TOP_N $CELLTYPE $HEAD head on $MODELS_DIR ensemble (distilled, mean+aleatoric+epistemic) exited with error code $exit_code"
		else
			slack "running $METHOD analysis for top $TOP_N $CELLTYPE on $MODELS_DIR ensemble (+ensemble avg of attr maps) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"