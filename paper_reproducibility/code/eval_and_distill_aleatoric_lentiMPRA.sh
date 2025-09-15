# run ensemble_predict_lentiMPRA.py with both --distill and --eval flags set 
# for ensemble of ResidualBind models with aleatoric uncertainty prediction


MODELS_DIR=../results/lentiMPRA_aleatoric # path to directory with ensemble of models 
N_MODS=10 # number of models in ensemble 
DATA_DIR=../data/lentiMPRA # path to directory with lentiMPRA data for models 

### boolean flags
EVOAUG=true # toggle true/false 
DOWNSAMPLED=false # toggle true/false

if [ "$EVOAUG" = true ]; then
	# update path to model directory
    MODELS_DIR=${MODELS_DIR}_evoaug
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5 # path to lentiMPRA data
MODELS_DIR=${MODELS_DIR}/${CELLTYPE} # update path to model directory

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=7 python ensemble_predict_lentiMPRA.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --aleatoric --evoaug --config ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/config.yaml
		else 
			# cannot use simple_gpu_scheduler bc running this script in parallel will result in errors due to multiple jobs reading from the same h5 file
			CUDA_VISIBLE_DEVICES=7 python ensemble_predict_lentiMPRA.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --aleatoric
		fi
	done 
else
	if [ "$EVOAUG" = true ]; then
		CUDA_VISIBLE_DEVICES=7 python ensemble_predict_lentiMPRA.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE --aleatoric --evoaug --config $MODELS_DIR/config.yaml
	else
		CUDA_VISIBLE_DEVICES=7 python ensemble_predict_lentiMPRA.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE --aleatoric
	fi 
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (downsampled, +evoaug) in distill and eval mode completed successfully" &>/dev/null
			else 	
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (downsampled) in distill and eval mode completed successfully" &>/dev/null
			fi
		else
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (full, +evoaug) in distill and eval mode completed successfully" &>/dev/null
			else 
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (full) in distill and eval mode completed successfully" &>/dev/null
			fi
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (downsampled, +evoaug) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (downsampled) in distill and eval mode exited with error code $exit_code"
			fi
		else	
			if [ "$EVOAUG" = true ]; then
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (full, +evoaug) in distill and eval mode exited with error code $exit_code"
			else 
				slack "running ensemble_predict_lentiMPRA.py for ResidualBind $CELLTYPE w/ aleatoric uncertainty prediction (full, +evoaug) in distill and eval mode exited with error code $exit_code"
			fi
		fi
	fi
fi
exit "$exit_code"
