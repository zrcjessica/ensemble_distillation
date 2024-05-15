# simultaneously evaluate prediction on test set and generate training data for distilled models
# for DeepSTARR models trained w/ EvoAug

DOWNSAMPLED=true # toggle true/false
# MODEL_DIR=../results/DeepSTARR_evoaug
MODEL_DIR=../results/DeepSTARR_evoaug/sanity_check
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		# echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		echo "python ensemble_predict_DeepSTARR.py \
					--model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} \
					--n_mods $N_MODS \
					--data $DATA \
					--distill \
					--eval \
					--plot \
					--downsample ${DOWNSAMPLE_ARR[$p]} \
					--evoaug \
					--config ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/config.yaml"
	done | simple_gpu_scheduler --gpus 1,2,3,4
else
	CUDA_VISIBLE_DEVICES=1 python ensemble_predict_DeepSTARR.py \
									--model_dir $MODEL_DIR \
									--n_mods $N_MODS \
									--data $DATA \
									--distill \
									--eval \
									--plot \
									--evoaug \
									--config ${MODEL_DIR}/config.yaml
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py for ${MODEL_DIR} (downsampled) in distill & eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_DeepSTARR.py for ${MODEL_DIR} (full) in distill & eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py for ${MODEL_DIR} (downsampled) in distill & eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_DeepSTARR.py for ${MODEL_DIR} (full) in distill & eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"