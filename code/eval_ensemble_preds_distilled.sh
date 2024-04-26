# runs ensemble_predict_DeepSTARR.py in eval mode only
# for distilled models

### bool flags
DOWNSAMPLED=false # toggle true/false to evaluate models trained on downsampled data
EVOAUG=true # toggle true/false for models trained w/ evoaug

### run variables
MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	MODEL_DIR=../results/DeepSTARR_evoaug
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	echo "evaluating performance of DeepSTARR ensemble trained on downsampled data"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		if [ "$EVOAUG" = true ]; then 
			echo "python ensemble_predict_DeepSTARR.py \
					--model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled \
					--n_mods $N_MODS \
					--data $DATA \
					--eval \
					--plot \
					--evoaug \
					--config ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled/config.yaml"
		else
			echo "python ensemble_predict_DeepSTARR.py \
					--model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled \
					--n_mods $N_MODS \
					--data $DATA \
					--eval \
					--plot"
		fi
	done | simple_gpu_scheduler --gpus 1,2,3,4
else
	if [ "$EVOAUG" = true ]; then 
		CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/ensemble_distilled --n_mods $N_MODS --data $DATA --eval --plot --evoaug --config ${MODEL_DIR}/ensemble_distilled/config.yaml
	else
		CUDA_VISIBLE_DEVICES=4,5 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/ensemble_distilled --n_mods $N_MODS --data $DATA --eval --plot 
	fi
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_DeepSTARR.py on distilled $MODEL_DIR (downsampled, trained with EvoAug) in eval mode completed successfully" &>/dev/null 
			else
				slack "running ensemble_predict_DeepSTARR.py on distilled $MODEL_DIR (downsampled) in eval mode completed successfully" &>/dev/null
			fi
		else
			if [ "$EVOAUG" = true ]; then 
				slack "running ensemble_predict_DeepSTARR.py for distilled $MODEL_DIR (full, trained with EvoAug) in eval mode completed successfully" &>/dev/null
			else	
				slack "running ensemble_predict_DeepSTARR.py for distilled $MODEL_DIR (full) in eval mode completed successfully" &>/dev/null
			fi
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_DeepSTARR.py on $MODEL_DIR (downsampled, trained with EvoAug) in eval mode exited with error code $exit_code"
			else
				slack "running ensemble_predict_DeepSTARR.py on $MODEL_DIR (downsampled) in eval mode exited with error code $exit_code"
			fi
		else
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_DeepSTARR.py on $MODEL_DIR (full, trained with EvoAug) in eval mode exited with error code $exit_code"
			else
				slack "running ensemble_predict_DeepSTARR.py on $MODEL_DIR (full) in eval mode exited with error code $exit_code"
			fi
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