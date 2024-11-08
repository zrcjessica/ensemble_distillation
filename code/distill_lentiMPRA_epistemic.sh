# train replicates of distilled ResidualBind models that predict mean+aleatoric+epistemic 

### script params/variables
N_MODS=10 # nr. of distilled models to train
MODELS_DIR=../results/distilled_lentiMPRA_epistemic # path to output directory
DATA_DIR=../data/lentiMPRA # path to directory with lentiMPRA data
CONFIG=../config/lentiMPRA.yaml # path to ResidualBind model config
PROJECT_NAME=lentiMPRA_distilled_epistemic # project name for WandB logging 

### boolean flags
downsample=true # train downsampled models
evoaug=true # train w/ evoaug
if [ "$evoaug" = true ]; then
    MODELS_DIR=${MODELS_DIR}_evoaug
    DATA_DIR=${DATA_DIR}/evoaug
fi
# use logvar instead of std for epistemic uncertainty 
logvar=false  
if [ "$logvar" = true ]; then  
    MODELS_DIR=${MODELS_DIR}/logvar
fi 

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}
DATA=${DATA_DIR}/${CELLTYPE}_distillation_data_with_epistemic.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    # DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) # used if downsample set to true
    DOWNSAMPLE_ARR=( 0.5 )
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
                     echo "echo 'model_ix=$i' && python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE --logvar"
                else 
                    echo "echo 'model_ix=$i' && python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
                fi
            else
                if [ "$logvar" = true ]; then
                    echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE --logvar"
                else 
                    echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DOWNSAMPLE_DATA --downsample ${DOWNSAMPLE_ARR[$p]} --plot  --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
                fi
            fi
        done | simple_gpu_scheduler --gpus 6
    done 
else
    OUTDIR=${MODELS_DIR}
    mkdir -p $OUTDIR
    for i in $(seq 1 $N_MODS)
    do 
        if [ "$evoaug" = true ]; then
            if [ "$logvar" = true ]; then
                echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE --logvar"
            else 
                echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
            fi
        else
            if [ "$logvar" = true ]; then
                echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE --logvar"
            else 
                echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
            fi
        fi
    done | simple_gpu_scheduler --gpus 6
fi


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled $N_MODS lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            else
		        slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE completed successfully" &>/dev/null
            else
                slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE completed successfully" &>/dev/null
            fi
        fi
	else
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            else
		        slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE exited with error code $exit_code"
            else
                slack "training $N_MODS replicates of distilled lentiMPRA models w/ epistemic uncertainty prediction for $CELLTYPE exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"