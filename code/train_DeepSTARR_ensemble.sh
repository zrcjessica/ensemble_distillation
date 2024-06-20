# train an ensemble of original DeepSTARR models 

ENSEMBLE_SIZE=10
# OUTDIR=../results/DeepSTARR_lr-decay
OUTDIR=../results/DeepSTARR_NEW
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
# PROJECT_NAME=DeepSTARR_ensemble
PROJECT_NAME=DeepSTARR_ensemble_NEW

# train w/ evoaug
evoaug=true
if [ "$evoaug" = true ]; then
    OUTDIR=../results/DeepSTARR_evoaug_NEW
    # PROJECT_NAME=DeepSTARR_ensemble_with_evoaug
fi

# train downsampled models 
downsample=true 

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do 
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        OUTDIR_DOWNSAMPLE=${OUTDIR}/downsample_${DOWNSAMPLE_ARR[$p]}
        mkdir -p $OUTDIR_DOWNSAMPLE
        for i in $(seq 1 $ENSEMBLE_SIZE)
        do
            if [ "$evoaug" = true ]; then
                echo "python train_DeepSTARR.py --ix $i --out ${OUTDIR_DOWNSAMPLE} --downsample ${DOWNSAMPLE_ARR[$p]} --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
            else
                echo "python train_DeepSTARR.py --ix $i --out ${OUTDIR_DOWNSAMPLE} --downsample ${DOWNSAMPLE_ARR[$p]} --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
            fi   
        done | simple_gpu_scheduler --gpus 6
    done 
else 
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
        else
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
        fi
    done | simple_gpu_scheduler --gpus 6
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$downsample" = true ]; then 
            if [ "$evoaug" = true ]; then
                slack "training ensemble of $ENSEMBLE_SIZE downsampled DeepSTARR models w/ EvoAug completed successfully" &>/dev/null
            else
                slack "training ensemble of $ENSEMBLE_SIZE downsampled DeepSTARR models completed successfully" &>/dev/null
            fi
        else 
            if [ "$evoaug" = true ]; then
                slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug completed successfully" &>/dev/null
            else
                slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models completed successfully" &>/dev/null
            fi
        fi 
	else
        if [ "$downsample" = true ]; then 
            if [ "$evoaug" = true ]; then
                slack "training ensemble of $ENSEMBLE_SIZE downsampled DeepSTARR models w/ EvoAug exited with error code $exit_code"
            else
                slack "training ensemble of $ENSEMBLE_SIZE downsampled DeepSTARR models exited with error code $exit_code"
            fi
        else 
            if [ "$evoaug" = true ]; then
                slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models w/ EvoAug exited with error code $exit_code"
            else
                slack "training ensemble of $ENSEMBLE_SIZE DeepSTARR models exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"