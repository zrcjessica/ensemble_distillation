# runs ensemble_predict_DeepSTARR.py in eval mode 

DOWNSAMPLED=false # toggle true/false to evaluate models trained on downsampled data
MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA 

if [ "$DOWNSAMPLED" = true ]; then
	echo "evaluating performance of DeepSTARR ensemble trained on downsampled data"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		for i in $(seq 1 $N_MODS)
		do 
			CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --eval
		done 
	done 
else
	CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA --eval
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (downsampled, trained with LR decay) in eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (full, trained with LR decay) in eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (downsampled, trained with LR decay) in eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (full, trained with LR decay) in eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"

# # message the user on slack if possible
# exit_code="$?"
# if command -v 'slack' &>/dev/null; then
#     if [ "$exit_code" -eq 0 ]; then
# 		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (full, trained with LR decay) completed successfully" &>/dev/null
# 	else
# 		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (full, trained with LR decay) exited with error code $exit_code"
# 	fi
# fi
# exit "$exit_code"