# train an ensemble distilled DeepSTARR model for downsample =0.25, model_ix=9 (originally failed)

N_MODS=10
# DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
MODELS_DIR=../results/DeepSTARR_lr-decay
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble # for wandb logger

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

ENSEMBLE_DIR=$MODELS_DIR/downsample_0.25
OUTDIR=$ENSEMBLE_DIR/ensemble_distilled_new
CUDA_VISIBLE_DEVICES=3 python train_DeepSTARR.py --ix 9 \
--out $OUTDIR --data $DATA --plot --config $CONFIG \
--project $PROJECT_NAME --lr_decay --distill $ENSEMBLE_DIR/ensemble_avg_y_train.npy \
--downsample 0.25


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training ensemble distilled model for model ix=9 and downsample p=0.25 from DeepSTARR_lr-decay (downsampled) completed successfully" &>/dev/null
	else
		slack "training ensemble distilled models for model ix=9 and downsample p=0.25 from DeepSTARR_lr-decay (downsampled) exited with error code $exit_code"
	fi
fi
exit "$exit_code"