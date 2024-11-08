# calculates stdev of ResidualBind (mean+aleatoric) ensemble predictions on train/test/val sets

CELLTYPE=K562 # cell type 
MODELS_DIR=../results/lentiMPRA_aleatoric/${CELLTYPE} # path to directory with ensemble of models 
OUTDIR=../data/lentiMPRA # path to output directory
N_MODS=10 # nr. of models in ensemble
CONFIG=../config/lentiMPRA.yaml # path to model config 
DATA=../data/lentiMPRA/${CELLTYPE}_data_with_aleatoric.h5 # path to lentiMPRA data 
# boolean vars (toggle true/false)
EVOAUG=true
DOWNSAMPLE=false

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	# update path to models and output directory
	MODELS_DIR=../results/lentiMPRA_aleatoric_evoaug/${CELLTYPE}
	OUTDIR=../data/lentiMPRA/evoaug
fi
mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
if [ "$DOWNSAMPLE" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=7 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --aleatoric
		else
			CUDA_VISIBLE_DEVICES=7 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --downsample ${DOWNSAMPLE_ARR[$p]} --celltype $CELLTYPE --aleatoric
		fi
	done
else
	if [ "$EVOAUG" = true ]; then
		CUDA_VISIBLE_DEVICES=7 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml --celltype $CELLTYPE --aleatoric
	else
		CUDA_VISIBLE_DEVICES=7 python get_lentiMPRA_ensemble_std.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --celltype $CELLTYPE --aleatoric
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