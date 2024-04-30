# run saliency analysis for top 500 Dev enhancers on an ensemble of DeepSTARR models
# set DISTILLED to perform attribution analysis for distilled models
# set METHOD as saliency or shap to define method of attribution analysis

DISTILLED=true # toggle flag
DOWNSAMPLED=true # toggle true/false
METHOD=saliency # set saliency or shap
N_MODS=10
MODELS_DIR=../results/DeepSTARR_evoaug
# if [ "$DISTILLED" = true ]; then MODELS_DIR=../results/DeepSTARR_evoaug/ensemble_distilled && echo "analyzing distilled models"; fi
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	echo "analyzing downsampled models"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"; do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		if [ "$DISTILLED" = true ]; then
			CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled --n_mods $N_MODS --data $DATA --method $METHOD --evoaug --config ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled/config.yaml
		else
			# typically only calculate average if looking at the ensemble, not the distilled models
			CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --method $METHOD --average --evoaug --config ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/config.yaml
		fi 
	done
else
	if [ "$DISTILLED" = true ]; then
		CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir ${MODELS_DIR}/ensemble_distilled --n_mods $N_MODS --data $DATA --method $METHOD --evoaug --config ${MODELS_DIR}/ensemble_distilled/config.yaml
	else
		# typically only calculate average if looking at the ensemble, not the distilled models
		CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --average --evoaug --config $MODELS_DIR/config.yaml
	fi 
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, trained with EvoAug) completed successfully" &>/dev/null
		else
			slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (no distillation, trained with EvoAug) completed successfully" &>/dev/null
		fi
	else
		if [ "$DISTILLED" = true ]; then
			slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, trained with EvoAug) exited with error code $exit_code"
		else
			slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} (no distillation, trained with EvoAug) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, downsampled trained with EvoAug) completed successfully" &>/dev/null 
            else 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (downsampled trained with EvoAug) completed successfully" &>/dev/null
            fi 
        else
            if [ "$DISTILLED" = true ]; then 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, trained with EvoAug) completed successfully" &>/dev/null 
            else 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (trained with EvoAug) completed successfully" &>/dev/null 
            fi 
        fi 
    else 
        if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, downsampled trained with EvoAug) exited with error code $exit_code" 
            else 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (downsampled trained with EvoAug) exited with error code $exit_code"
            fi 
        else
            if ["$DISTILLED" = true ]; then 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (distilled, trained with EvoAug) exited with error code $exit_code"
            else 
                slack "running $METHOD analysis for top 500 Dev on ${MODELS_DIR} ensemble (trained with EvoAug) exited with error code $exit_code"
            fi 
        fi 
    fi
fi
exit "$exit_code"