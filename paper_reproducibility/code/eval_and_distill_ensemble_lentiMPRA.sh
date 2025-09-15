# run ensemble_predict_lentiMPRA.py with both --distill and --eval flags set
# evaluates performance of ensemble average and returns ensemble average on training seqs


MODELS_DIR=../results/lentiMPRA # path to directory containing teacher models 
N_MODS=10 # number of teacher models in ensemble 
DATA_DIR=../data/lentiMPRA # path to directory containing lentiMPRA data 

### boolean flags
EVOAUG=false # toggle true/false 
DOWNSAMPLED=true # toggle true/false

# adjust path to teacher models dir
if [ "$EVOAUG" = true ]; then
    MODEL_DIR=${MODEL_DIR}/evoaug
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
DATA=${DATA_DIR}/${CELLTYPE}_data.h5 # path to lentiMPRA data for cell type 
MODELS_DIR=${MODELS_DIR}/${CELLTYPE} # path to teacher models dir

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$DOWNSAMPLED" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
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
			slack "running ensemble_predict_lentiMPRA.py for ResidualBind (downsampled) in distill and eval mode completed successfully" &>/dev/null
		else
			slack "running ensemble_predict_lentiMPRA.py for ResidualBind (full) in distill and eval mode completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLED" = true ]; then
			slack "running ensemble_predict_lentiMPRA.py for ResidualBind (downsampled) in distill and eval mode exited with error code $exit_code"
		else
			slack "running ensemble_predict_lentiMPRA.py for ResidualBind (full) in distill and eval mode exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"
