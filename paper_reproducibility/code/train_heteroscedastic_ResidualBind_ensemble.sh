# train an ensemble of ResidualBind models with heteroscedastic regression 

### script params/variables
ENSEMBLE_SIZE=10
# ENSEMBLE_SIZE=20    
OUTDIR=../results/ResidualBind_heteroscedastic
DATA_DIR=../data/lentiMPRA
CONFIG=../config/lentiMPRA.yaml
PROJECT_NAME=ResidualBind_heteroscedastic_logvar
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) # used if downsample set to true
### define cell type
CELLTYPE='HepG2' # HepG2 or K562
OUTDIR=${OUTDIR}/${CELLTYPE}
DATA=${DATA_DIR}/${CELLTYPE}_data_with_aleatoric.h5

### boolean flags
# train downsampled models
downsample=false
# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    OUTDIR=${OUTDIR}_evoaug
fi

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
                echo "echo 'model_ix=$i' && python train_heteroscedastic_ResidualBind.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr 0.001 --lr_decay --evoaug --celltype $CELLTYPE"
            else
                echo "echo 'model_ix=$i' && python train_heteroscedastic_ResidualBind.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr 0.001 --lr_decay --celltype $CELLTYPE"
            fi
        done | simple_gpu_scheduler --gpus 0,3,4,5,6
    done 
else
    for i in $(seq 1 $ENSEMBLE_SIZE)
    # for i in $(seq 11 $ENSEMBLE_SIZE)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_heteroscedastic_ResidualBind.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --lr 0.001 --evoaug --celltype $CELLTYPE"
        else
            echo "python train_heteroscedastic_ResidualBind.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --lr 0.001 --celltype $CELLTYPE"
        fi
    done | simple_gpu_scheduler --gpus 0,3,4,5,6
fi


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE completed successfully" &>/dev/null
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE completed successfully" &>/dev/null
            fi
        fi
	else
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression with for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training ensemble of downsampled $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE exited with error code $exit_code"
            else
                slack "training ensemble of $ENSEMBLE_SIZE ResidualBind models with heteroscedastic regression for $CELLTYPE exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"