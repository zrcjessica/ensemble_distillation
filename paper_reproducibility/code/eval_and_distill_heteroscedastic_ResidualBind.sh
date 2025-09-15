# run ensemble_predict_heteroscedastic_ResidualBind.py with both --distill and --eval flags set

# runs ensemble_predict_heteroscedastic_ResidualBind.py in distill mode
# for ensemble of ResidualBind models trained with heteroscedastic regression
# toggle DOWNSAMPLED to control whether distilled training data is obtained for downsampled models


MODELS_DIR=../results/ResidualBind_heteroscedastic
N_MODS=10
DATA_DIR=../data/lentiMPRA
### boolean flags
EVOAUG=false 
DOWNSAMPLED=false 
# ALEATORIC=true

if [ "$EVOAUG" = true ]; then
    MODELS_DIR=${MODELS_DIR}_evoaug
fi 

# if [ "$ALEATORIC" = true ]; then 
# 	MODELS_DIR=${MODELS_DIR}_aleatoric
# fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}
CONFIG=${MODELS_DIR}/config.yaml

# if [ "$ALEATORIC" = true ]; then 
# 	DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5
# fi 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		mkdir -p ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
		CUDA_VISIBLE_DEVICES=0 python ensemble_predict_heteroscedastic_ResidualBind.py --model_dir ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} \
		--n_mods $N_MODS --data $DATA --distill --eval --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE \
		--out ${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG
	done 
else
	CUDA_VISIBLE_DEVICES=0 python ensemble_predict_heteroscedastic_ResidualBind.py --model_dir $MODELS_DIR \
	--n_mods $N_MODS --data $DATA --distill --eval --celltype $CELLTYPE --plot --config $CONFIG
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_heteroscedastic_ResidualBind.py for ResidualBind $CELLTYPE (downsampled) in distill and eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_heteroscedastic_ResidualBind.py for ResidualBind $CELLTYPE (full) with aleatoric uncertainty in distill and eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_heteroscedastic_ResidualBind.py for ResidualBind $CELLTYPE (downsampled) with aleatoric uncertainty in distill and eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_heteroscedastic_ResidualBind.py for ResidualBind $CELLTYPE (full) in distill and eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"
