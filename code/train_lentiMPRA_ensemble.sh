# train an ensemble of ResidualBind models with train_lentiMPRA.py

### define variables
ENSEMBLE_SIZE=10 # number of models to train
OUTDIR=../results/lentiMPRA # path to output directory
DATA_DIR=../data/lentiMPRA # path to lentiMPRA data
CONFIG=../config/lentiMPRA.yaml # path to ResidualBind model config
PROJECT_NAME=lentiMPRA_ensemble # project name for WandB logging
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) # used if downsample set to true

### boolean flags
downsample=true # train downsampled models
evoaug=false # train w/ evoaug
if [ "$evoaug" = true ]; then
    # modify output directory
    OUTDIR=${OUTDIR}_evoaug
fi

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
OUTDIR=${OUTDIR}/${CELLTYPE}
DATA=${DATA_DIR}/${CELLTYPE}_data.h5

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        OUTDIR_DOWNSAMPLE=${OUTDIR}/downsample_${DOWNSAMPLE_ARR[$p]}
        mkdir -p $OUTDIR_DOWNSAMPLE
        for i in $(seq 1 $ENSEMBLE_SIZE)
        do 
            if [ "$evoaug" = true ]; then
                echo "echo 'model_ix=$i' && python train_lentiMPRA.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
            else
                echo "echo 'model_ix=$i' && python train_lentiMPRA.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
            fi
        done | simple_gpu_scheduler --gpus 1,2,3,4,5
    done 
else
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_lentiMPRA.py --ix $i --out $OUTDIR --data $DATA --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
        else
            echo "python train_lentiMPRA.py --ix $i --out $OUTDIR --data $DATA --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
        fi
    done | simple_gpu_scheduler --gpus 0,1,2,3,4
fi


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            else
		        slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE completed successfully" &>/dev/null
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE completed successfully" &>/dev/null
            fi
        fi
	else
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            else
		        slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE exited with error code $exit_code"
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models for $CELLTYPE exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"