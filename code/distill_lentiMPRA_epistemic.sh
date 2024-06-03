# train replicates of distilled lentiMPRA models w/ mean+epistemic+epistemic predictions

### script params/variables
N_MODS=10
MODELS_DIR=../results/distilled_lentiMPRA_epistemic
DATA_DIR=../data/lentiMPRA
CONFIG=../config/lentiMPRA.yaml
PROJECT_NAME=lentiMPRA_distilled_epistemic

### boolean flags
# train downsampled models
downsample=false
# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    MODELS_DIR=../results/distilled_lentiMPRA_epistemic-evoaug
fi

### define cell type
CELLTYPE='K562' # HepG2 or K562
MODELS_DIR=${MODELS_DIR}/${CELLTYPE}
DATA=${DATA_DIR}/${CELLTYPE}_distillation_data_with_epistemic.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) # used if downsample set to true
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        OUTDIR_DOWNSAMPLE=${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled
        mkdir -p $OUTDIR_DOWNSAMPLE
        for i in $(seq 1 $N_MODS)
        do 
            if [ "$evoaug" = true ]; then
                echo "echo 'model_ix=$i' && python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
            else
                echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
            fi
        done | simple_gpu_scheduler --gpus 5,6
    done 
else
    OUTDIR=${MODELS_DIR}/ensemble_distilled
    mkdir -p $OUTDIR
    for i in $(seq 1 $N_MODS)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --celltype $CELLTYPE"
        else
            echo "python train_distilled_lentiMPRA_with_epistemic.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --celltype $CELLTYPE"
        fi
    done | simple_gpu_scheduler --gpus 5,6
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