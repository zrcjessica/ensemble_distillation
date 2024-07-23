# run ensemble_predict_MPRAnn.py with both --distill and --eval flags set

# runs ensemble_predict_MPRAnn.py in distill mode
# for MPRAnn ensemble trained on full training data
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models


MODELS_DIR=../results/MPRAnn
N_MODS=10
DATA_DIR=../data/lentiMPRA

### boolean flags
EVOAUG=false 
DOWNSAMPLED=true 
ALEATORIC=true

if [ "$EVOAUG" = true ]; then
    MODELS_DIR=${MODELS_DIR}/evoaug
fi 

if [ "$ALEATORIC" = true ]; then 
	MODELS_DIR=${MODELS_DIR}_aleatoric
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data.h5
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}

if [ "$ALEATORIC" = true ]; then 
	DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5
fi 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		mkdir -p ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
		if [ "$ALEATORIC" = true ]; then 
			CUDA_VISIBLE_DEVICES=1 python ensemble_predict_MPRAnn.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --out ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --aleatoric
		else 
			CUDA_VISIBLE_DEVICES=1 python ensemble_predict_MPRAnn.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --out ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
		fi 
	done 
else
	if [ "$ALEATORIC" = true ]; then 
		CUDA_VISIBLE_DEVICES=1 python ensemble_predict_MPRAnn.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE --aleatoric 
	else 
		CUDA_VISIBLE_DEVICES=1 python ensemble_predict_MPRAnn.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE
	fi 
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$ALEATORIC" = true ]; then 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (downsampled) with aleatoric uncertainty in distill and eval mode completed successfully" &>/dev/null
			else 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (downsampled) in distill and eval mode completed successfully" &>/dev/null
			fi 
		else
			if [ "$ALEATORIC" = true ]; then 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (full) in distill and eval mode completed successfully" &>/dev/null
			else 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (full) with aleatoric uncertainty in distill and eval mode completed successfully" &>/dev/null
			fi 
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$ALEATORIC" = true ]; then 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (downsampled) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (downsampled) with aleatoric uncertainty in distill and eval mode exited with error code $exit_code"
			fi 
		else
			if [ "$ALEATORIC" = true ]; then 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (full) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_MPRAnn.py for MPRAnn (full) in distill and eval mode exited with error code $exit_code"
			fi 
		fi
	fi
fi
exit "$exit_code"
