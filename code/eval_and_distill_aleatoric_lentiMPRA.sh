# run ensemble_predict_lentiMPRA.py with both --distill and --eval flags set

# runs ensemble_predict_lentiMPRA.py in distill mode
# for lentiMPRA ensemble trained on full training data w/ aleatoric uncertainty prediction 
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models


MODELS_DIR=../results/lentiMPRA_aleatoric
N_MODS=10
DATA_DIR=../data/lentiMPRA

### boolean flags
EVOAUG=false # toggle true/false 
DOWNSAMPLED=false # toggle true/false

if [ "$EVOAUG" = true ]; then
    MODEL_DIR=${MODEL_DIR}/evoaug
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		# cannot use simple_gpu_scheduler bc running this script in parallel will result in errors due to multiple jobs reading from the same h5 file
		CUDA_VISIBLE_DEVICES=5 python ensemble_predict_lentiMPRA.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --aleatoric
	done 
else
	CUDA_VISIBLE_DEVICES=5 python ensemble_predict_lentiMPRA.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE --aleatoric
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA $CELLTYPE w/ aleatoric uncertainty prediction (downsampled) in distill and eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA $CELLTYPE w/ aleatoric uncertainty prediction (full) in distill and eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA $CELLTYPE w/ aleatoric uncertainty prediction (downsampled) in distill and eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_lentiMPRA.py for lentiMPRA $CELLTYPE w/ aleatoric uncertainty prediction (full) in distill and eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"
