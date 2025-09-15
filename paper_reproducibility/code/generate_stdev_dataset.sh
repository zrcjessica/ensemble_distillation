# calculates stdev of DeepSTARR ensemble predictions on train/test/val

# MODELS_DIR=../results/DeepSTARR_lr-decay
MODELS_DIR=../results/DeepSTARR_ensemble_NEW # path to ensemble of teacher models 
N_MODS=10 # nr. of models in ensemble 
DATA=../data/DeepSTARR/Sequences_activity_all.h5 # path to STARR-seq data
OUTDIR=../data/DeepSTARR_ensemble_NEW # where to save output files

# boolean vars (toggle true/false)
EVOAUG=true
DOWNSAMPLE=true 

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	MODELS_DIR=../results/DeepSTARR_evoaug_NEW
	OUTDIR=../data/DeepSTARR_evoaug
fi

mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
if [ "$DOWNSAMPLE" = true ]; then
	# analyze downsampled models
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		if [ "$EVOAUG" = true ]; then
			CUDA_VISIBLE_DEVICES=7 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml --downsample ${DOWNSAMPLE_ARR[$p]}
		else
			CUDA_VISIBLE_DEVICES=7 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --downsample ${DOWNSAMPLE_ARR[$p]}
		fi
	done
else
	if [ "$EVOAUG" = true ]; then
		CUDA_VISIBLE_DEVICES=7 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml
	else
		CUDA_VISIBLE_DEVICES=7 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR
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