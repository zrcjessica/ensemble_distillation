# train an ensemble of DeepSTARR models on downsampled training data
# loops through an array of different downsampling proportions

ENSEMBLE_SIZE=10
# DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
DOWNSAMPLE_ARR=( 0.75 )
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble
OUTDIR_ROOT=../results/DeepSTARR_lr-decay

# train w/ evoaug
evoaug=true
if [ "$evoaug" = true ]; then
    OUTDIR_ROOT=../results/DeepSTARR_lr-decay/evoaug
    # PROJECT_NAME=DeepSTARR_ensemble_with_evoaug
fi


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
    OUTDIR=$OUTDIR_ROOT/downsample_${DOWNSAMPLE_ARR[$p]}
    mkdir -p $OUTDIR
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
        else
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay"
        fi
    done | simple_gpu_scheduler --gpus 5
done 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training downsampled ensembles of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug (p=0.75) completed successfully" &>/dev/null
	else
		slack "training downsampled ensembles of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug exited with error code $exit_code"
	fi
fi
exit "$exit_code"