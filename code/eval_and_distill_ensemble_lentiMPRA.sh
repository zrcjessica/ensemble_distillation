# run ensemble_predict_lentiMPRA.py with both --distill and --eval flags set

# runs ensemble_predict_lentiMPRA.py in distill mode
# for lentiMPRA ensemble trained on full training data
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models


MODELS_DIR=../results/lentiMPRA
N_MODS=10
DATA_DIR=../data/lentiMPRA

### boolean flags
EVOAUG=false # toggle true/false 
DOWNSAMPLED=true # toggle true/false

if [ "$EVOAUG" = true ]; then
    MODEL_DIR=${MODEL_DIR}/evoaug
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data.h5
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	# DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	DOWNSAMPLE_ARR=( 0.1 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		# cannot use simple_gpu_scheduler bc running this script in parallel will result in errors due to multiple jobs reading from the same h5 file
		# CUDA_VISIBLE_DEVICES=1 python ensemble_predict_lentiMPRA.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE
		mkdir -p ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/20mods
		CUDA_VISIBLE_DEVICES=2 python ensemble_predict_lentiMPRA.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --out ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/20mods
	done 
else
	CUDA_VISIBLE_DEVICES=1 python ensemble_predict_lentiMPRA.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA (downsampled) in distill and eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA (full) in distill and eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA (downsampled) in distill and eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA (full) in distill and eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"

# # message the user on slack if possible
# exit_code="$?"
# if command -v 'slack' &>/dev/null; then
#     if [ "$exit_code" -eq 0 ]; then
# 		slack "running ensemble_predict_lentiMPRA.py for lentiMPRA_lr-decay (no downsampling) to distill training data completed successfully" &>/dev/null
# 	else
# 		slack "running ensemble_predict_lentiMPRA.py for lentiMPRA_lr-decay (no downsampling) to distill training data exited with error code $exit_code"
# 	fi
# fi
# exit "$exit_code"