# train an ensemble distilled DeepSTARR model using distilled training data

ENSEMBLE_SIZE=10
ENSEMBLE_DIR=../results/DeepSTARR_lr-decay
OUTDIR=$ENSEMBLE_DIR/ensemble_distilled
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble # for wandb logger

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for i in $(seq 1 $ENSEMBLE_SIZE)
do 
    echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --distill $ENSEMBLE_DIR/distilled_y_train.npy"
done | simple_gpu_scheduler --gpus 4,5,6,7


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training ensemble distilled models (n=${ENSMBLE_SIZE}) from DeepSTARR_lr-decay (full) completed successfully" &>/dev/null
	else
		slack "training ensemble distilled models (n=${ENSMBLE_SIZE}) from DeepSTARR_lr-decay (full) exited with error code $exit_code"
	fi
fi
exit "$exit_code"