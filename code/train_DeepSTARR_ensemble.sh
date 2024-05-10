# train an ensemble of original DeepSTARR models 

ENSEMBLE_SIZE=10
OUTDIR=../results/DeepSTARR_lr-decay
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble

# train w/ evoaug
evoaug=true
if [ "$evoaug" = true ]; then
    OUTDIR=../results/DeepSTARR_evoaug
    # PROJECT_NAME=DeepSTARR_ensemble_with_evoaug
fi

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


for i in $(seq 1 $ENSEMBLE_SIZE)
do 
    if [ "$evoaug" = true ]; then
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
    else
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
    fi
done | simple_gpu_scheduler --gpus 3,4,7


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug completed successfully" &>/dev/null
	else
		slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug exited with error code $exit_code"
	fi
fi
exit "$exit_code"