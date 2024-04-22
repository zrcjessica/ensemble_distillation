# train an ensemble distilled DeepSTARR model for all downsampled ensembles using distilled training data

N_MODS=10
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
MODELS_DIR=../results/DeepSTARR_lr-decay
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble # for wandb logger

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

### NOTE: how do I run this with simple gpu scheduler? 
for i in $(seq 1 $N_MODS)
do
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        ENSEMBLE_DIR=$MODELS_DIR/downsample_${DOWNSAMPLE_ARR[$p]}
        OUTDIR=$ENSEMBLE_DIR/ensemble_distilled_new
        mkdir -p $OUTDIR
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --distill $ENSEMBLE_DIR/ensemble_avg_y_train.npy --downsample ${DOWNSAMPLE_ARR[$p]}"
    done | simple_gpu_scheduler --gpus 4,5,6,7 
done 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training ensemble distilled models (n=${N_MODS}) from DeepSTARR_lr-decay (downsampled) completed successfully" &>/dev/null
	else
		slack "training ensemble distilled models (n=${N_MODS}) from DeepSTARR_lr-decay (downsampled) exited with error code $exit_code"
	fi
fi
exit "$exit_code"