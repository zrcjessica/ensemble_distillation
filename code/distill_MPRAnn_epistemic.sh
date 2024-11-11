# train replicates of distilled MPRAnn models w/ activity+epistemic+epistemic predictions

### script params/variables
N_MODS=10 # nr. models to train 
MODELS_DIR=../results/distilled_MPRAnn_epistemic # path to output dir
DATA_DIR=../data/MPRAnn_aleatoric # path to directory with data 
CONFIG=../config/MPRAnn.yaml # path to MPRAnn model config 
PROJECT_NAME=MPRAnn_distilled_epistemic # project name for WandB logging 

### boolean flags
downsample=false # train downsampled models
evoaug=false # train w/ evoaug
if [ "$evoaug" = true ]; then
    # update paths 
    MODELS_DIR=${MODELS_DIR}_evoaug
    DATA_DIR=${DATA_DIR}/evoaug
fi
logvar=false  # use logvar instead of std for epistemic uncertainty 
if [ "$logvar" = true ]; then  
    # update paths 
    MODELS_DIR=${MODELS_DIR}/logvar
fi 
heteroscedastic=true # teacher models trained with heteroscedastic regression
if [ "$heteroscedastic" = true ]; then 
    # update paths
    DATA_DIR=../data/MPRAnn_heteroscedastic
    MODELS_DIR=../results/distilled_MPRAnn_heteroscedastic
    PROJECT_NAME=MPRAnn_heteroscedastic_distilled
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
MODELS_DIR=${MODELS_DIR}/${CELLTYPE} # update paths 
DATA=${DATA_DIR}/${CELLTYPE}_distillation_data_with_epistemic.h5 # path to data HDF5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) # used if downsample set to true
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        OUTDIR=${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
        DOWNSAMPLE_DATA=${DATA_DIR}/${CELLTYPE}_downsample${DOWNSAMPLE_ARR[$p]}_distillation_data_with_epistemic.h5
        mkdir -p $OUTDIR
        for i in $(seq 1 $N_MODS)
        do 
            if [ "$evoaug" = true ]; then
                if [ "$logvar" = true ]; then
                     echo "echo 'model_ix=$i' && python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE --logvar"
                else 
                    echo "echo 'model_ix=$i' && python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
                fi
            else
                if [ "$logvar" = true ]; then
                    echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE --logvar"
                else 
                    echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot  --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
                fi
            fi
        done | simple_gpu_scheduler --gpus 0,1,2,3
    done 
else
    OUTDIR=${MODELS_DIR}
    mkdir -p $OUTDIR
    # for i in $(seq 1 $N_MODS)
    for i in $(seq 1 9)
    do 
        if [ "$evoaug" = true ]; then
            if [ "$logvar" = true ]; then
                echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE --logvar"
            else 
                echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
            fi
        else
            if [ "$logvar" = true ]; then
                echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE --logvar"
            else 
                echo "python train_distilled_MPRAnn_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
            fi
        fi
    done | simple_gpu_scheduler --gpus 0,1,2,3
fi


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled $N_MODS MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            else
		        slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE completed successfully" &>/dev/null
            else
                slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE completed successfully" &>/dev/null
            fi
        fi
	else
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            else
		        slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE exited with error code $exit_code"
            else
                slack "training $N_MODS replicates of distilled MPRAnn models w/ epistemic uncertainty prediction for $CELLTYPE exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"