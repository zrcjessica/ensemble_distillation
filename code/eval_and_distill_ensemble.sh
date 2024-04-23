# run ensemble_predict_DeepSTARR.py with both --distill and --eval flags set

# runs ensemble_predict_DeepSTARR.py in distill mode
# for DeepSTARR ensemble trained on full training data
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models

DOWNSAMPLED=true # toggle true/false
MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5
EVOAUG=true # toggle true/false 

if [ "$EVOAUG" = true ]; then
    MODEL_DIR=${MODEL_DIR}/evoaug
fi 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# CUDA_VISIBLE_DEVICES=4,5 python get_train_data_for_distillation.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA

if [ "$DOWNSAMPLED" = true ]; then
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		CUDA_VISIBLE_DEVICES=4 python ensemble_predict_DeepSTARR.py --model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} 
	done 
else
	CUDA_VISIBLE_DEVICES=4 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA --distill --eval
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (downsampled) in distill and eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (full) in distill and eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (downsampled) in distill and eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (full) in distill and eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"

# # message the user on slack if possible
# exit_code="$?"
# if command -v 'slack' &>/dev/null; then
#     if [ "$exit_code" -eq 0 ]; then
# 		slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (no downsampling) to distill training data completed successfully" &>/dev/null
# 	else
# 		slack "running ensemble_predict_DeepSTARR.py for DeepSTARR_lr-decay (no downsampling) to distill training data exited with error code $exit_code"
# 	fi
# fi
# exit "$exit_code"