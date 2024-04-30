# calculates stdev of DeepSTARR ensemble predictions on train/test/val

EVOAUG=true
MODELS_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5
OUTDIR=../data/DeepSTARR

# flag dependent changes
if [ "$EVOAUG" = true ]; then
	MODELS_DIR=../results/DeepSTARR_evoaug
	OUTDIR=../data/DeepSTARR/evoaug
fi

mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$EVOAUG" = true ]; then
	CUDA_VISIBLE_DEVICES=4,5 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR --evoaug --config $MODELS_DIR/config.yaml
else
	CUDA_VISIBLE_DEVICES=4,5 python stdev_ensemble_predictions.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --out $OUTDIR
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "getting train/test/val ensemble stdev for $MODELS_DIR (full) completed successfully" &>/dev/null
	else
		slack "getting train/test/val ensemble stdev for $MODELS_DIR (full) exited with error code $exit_code"
	fi
fi
exit "$exit_code"