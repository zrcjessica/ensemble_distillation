# run ensemble_predict_DeepSTARR.py with both --distill and --eval flags set
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models

DOWNSAMPLED=true # toggle true/false
# MODEL_DIR=../results/DeepSTARR_lr-decay/sanity_check
MODEL_DIR=../results/DeepSTARR_ensemble_NEW # path to ensemble of teacher DeepSTARR models
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5 # path to STARR-seq data
EVOAUG=true # toggle true/false 

if [ "$EVOAUG" = true ]; then
	MODEL_DIR=../results/DeepSTARR_evoaug_NEW
fi 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} \
			--n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} \
			--evoaug --config ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/config.yaml
		else
			CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} \
			--n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]}
		fi 
	done 
else
	if [ "$EVOAUG" = true ]; then 
		CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA --distill --eval --evoaug --config ${MODEL_DIR}/config.yaml
	else 
		CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA --distill --eval
	fi
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (downsampled) in distill and eval mode completed successfully" &>/dev/null
			else 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (downsampled +evoaug) in distill and eval mode completed successfully" &>/dev/null
			fi 
		else
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (full +evoaug) in distill and eval mode completed successfully" &>/dev/null
			else 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (full +evoaug) in distill and eval mode completed successfully" &>/dev/null
			fi
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (downsampled +evoaug) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (downsampled +evoaug) in distill and eval mode exited with error code $exit_code"
			fi 
		else
			if [ "$EVOAUG" = true ]; then 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (full +evoaug) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_DeepSTARR.py for $MODEL_DIR (full +evoaug) in distill and eval mode exited with error code $exit_code"
			fi
		fi
	fi
fi
exit "$exit_code"
