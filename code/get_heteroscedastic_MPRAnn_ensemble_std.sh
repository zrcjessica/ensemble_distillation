# calculates stdev of MPRAnn model (activity+aleatoric) ensemble predictions on train/test/val sets

CELLTYPE=HepG2
MODELS_DIR=../results/MPRAnn_heteroscedastic/${CELLTYPE}
OUTDIR=../data/MPRAnn_heteroscedastic
N_MODS=10
DATA=../data/lentiMPRA/${CELLTYPE}_data_with_aleatoric.h5
CONFIG=${MODELS_DIR}/config.yaml

# boolean flags
EVOAUG=false
DOWNSAMPLE=false

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	MODELS_DIR=../results/MPRAnn_heteroscedastic_evoaug/${CELLTYPE}
	OUTDIR=../data/MPRAnn_heteroscedastic/evoaug
fi

mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
if [ "$DOWNSAMPLE" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=0 python get_heteroscedastic_MPRAnn_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $CONFIG --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE 
		else
			CUDA_VISIBLE_DEVICES=0 python get_heteroscedastic_MPRAnn_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --config $CONFIG
		fi
	done
else
	if [ "$EVOAUG" = true ]; then
		CUDA_VISIBLE_DEVICES=0 python get_heteroscedastic_MPRAnn_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $CONFIG --celltype $CELLTYPE 
	else
		CUDA_VISIBLE_DEVICES=0 python get_heteroscedastic_MPRAnn_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --celltype $CELLTYPE --config $CONFIG
	fi
fi 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLE" = true ]; then
			slack "getting train/test/val ensemble stdev for $MODELS_DIR (downsampled) completed successfully" &>/dev/null
		else
			slack "getting train/test/val ensemble stdev for $MODELS_DIR (full) completed successfully" &>/dev/null
		fi
	else
		if [ "$DOWNSAMPLE" = true ]; then
			slack "getting train/test/val ensemble stdev for $MODELS_DIR (dowwnsampled) exited with error code $exit_code"
		else 
			slack "getting train/test/val ensemble stdev for $MODELS_DIR (full) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"