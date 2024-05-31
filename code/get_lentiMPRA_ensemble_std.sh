# calculates stdev of lentiMPRA (mean+aleatoric) ensemble predictions for activity level on train/test/val sets

CELLTYPE=K562
MODELS_DIR=../results/lentiMPRA_aleatoric/${CELLTYPE}
OUTDIR=../data/lentiMPRA/${CELLTYPE}
N_MODS=10
CONFIG=../config/lentiMPRA.yaml

# boolean vars (toggle true/false)
EVOAUG=false
DOWNSAMPLE=false

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	MODELS_DIR=../results/lentiMPRA_aleatoric-evoaug/${CELLTYPE}
	OUTDIR=../data/lentiMPRA/evoaug/${CELLTYPE}
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
if [ "$DOWNSAMPLE" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=4 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE
		else
			CUDA_VISIBLE_DEVICES=4 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE
		fi
	done
else
	if [ "$EVOAUG" = true ]; then
		CUDA_VISIBLE_DEVICES=4 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml --celltype $CELLTYPE
	else
		CUDA_VISIBLE_DEVICES=4 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --celltype $CELLTYPE
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