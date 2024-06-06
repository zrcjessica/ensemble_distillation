# train replicates of DeepSTARR models w/ evidential regression

ENSEMBLE_SIZE=10
OUTDIR=../results/DeepSTARR_evidential
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_evidential

# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    OUTDIR=${OUTDIR}_evoaug
    # PROJECT_NAME=DeepSTARR_ensemble_with_evoaug
fi

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


for i in $(seq 1 $ENSEMBLE_SIZE)
do 
    if [ "$evoaug" = true ]; then
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --evidential"
    else
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evidential"
    fi
done | simple_gpu_scheduler --gpus 1,2,3,4,5


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ evidential regression completed successfully" &>/dev/null
	else
		slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ evidential regression exited with error code $exit_code"
	fi
fi
exit "$exit_code"