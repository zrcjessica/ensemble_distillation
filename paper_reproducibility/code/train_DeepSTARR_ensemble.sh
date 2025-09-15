# train an ensemble of DeepSTARR models with standard training 

ENSEMBLE_SIZE=10 # define number of models to trian 
# ENSEMBLE_SIZE=25   
# OUTDIR=../results/DeepSTARR_lr-decay
OUTDIR=../results/DeepSTARR_ensemble_NEW # define output directory
# OUTDIR=../results/DeepSTARR_ensemble_size
DATA=../data/DeepSTARR/Sequences_activity_all.h5 # path to data 
CONFIG=../config/DeepSTARR.yaml # path to model config 
# PROJECT_NAME=DeepSTARR_ensemble_size
PROJECT_NAME=DeepSTARR_ensemble_NEW # name of project for logging with WandB

# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    OUTDIR=../results/DeepSTARR_evoaug_NEW
fi

# train downsampled models 
downsample=false 

# create output dir if it doesn't already exist 
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
    # for i in $(seq 11 $ENSEMBLE_SIZE)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
        else
            echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
        fi
    done | simple_gpu_scheduler --gpus 0,1,2,3,4,5
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