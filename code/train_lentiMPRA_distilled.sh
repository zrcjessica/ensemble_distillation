# train replicates of distilled DeepSTARR models using distilled training data

### script params/variables
# ENSEMBLE_SIZE=3
N_MODS=10
ENSEMBLE_DIR=../results/lentiMPRA
DATA_DIR=../data/lentiMPRA
CONFIG=../config/lentiMPRA.yaml
# PROJECT_NAME=lentiMPRA_distill_from_ranked_ensemble
PROJECT_NAME=lentiMPRA_ensemble

### boolean flags
# train downsampled models
downsample=true
# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    ENSEMBLE_DIR=../results/lentiMPRA-evoaug
fi

### define cell type
CELLTYPE='HepG2' # HepG2 or K562
ENSEMBLE_DIR=${ENSEMBLE_DIR}/${CELLTYPE}
DATA=${DATA_DIR}/${CELLTYPE}_data.h5
	
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
    echo 'training on downsampled data'
    # DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 ) 
    DOWNSAMPLE_ARR=( 0.1 )
    for p in "${!DOWNSAMPLE_ARR[@]}"
    do
        echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        # OUTDIR_DOWNSAMPLE=${ENSEMBLE_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled
        OUTDIR_DOWNSAMPLE=${ENSEMBLE_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/20mods/ensemble_distilled
        mkdir -p $OUTDIR_DOWNSAMPLE
        for i in $(seq 1 $N_MODS)
        do 
            if [ "$evoaug" = true ]; then
                echo "echo 'model_ix=$i' && python train_lentiMPRA.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --distill ${ENSEMBLE_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_avg_y_train.npy --lr_decay --evoaug --celltype $CELLTYPE"
            else
                # echo "python train_lentiMPRA.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --distill ${ENSEMBLE_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_avg_y_train.npy --lr_decay --celltype $CELLTYPE"
                echo "python train_lentiMPRA.py --ix $i --out $OUTDIR_DOWNSAMPLE --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME --distill ${ENSEMBLE_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/20mods/ensemble_avg_y_train.npy --lr_decay --celltype $CELLTYPE"
            fi
        done | simple_gpu_scheduler --gpus 5,6
    done 
else
    OUTDIR=${ENSEMBLE_DIR}/ensemble_distilled
    mkdir -p $OUTDIR
    for i in $(seq 1 $N_MODS)
    do 
        if [ "$evoaug" = true ]; then
            echo "python train_lentiMPRA.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --distill ${ENSEMBLE_DIR}/ensemble_avg_y_train.npy --evoaug --celltype $CELLTYPE"
        else
            echo "python train_lentiMPRA.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --distill ${ENSEMBLE_DIR}/ensemble_avg_y_train.npy --celltype $CELLTYPE"
        fi
    done | simple_gpu_scheduler --gpus 0,1,2,3,4
fi


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled lentiMPRA models for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            else
		        slack "training $N_MODS replicates of distilled lentiMPRA models for $CELLTYPE w/ EvoAug completed successfully" &>/dev/null
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled lentiMPRA models for $CELLTYPE completed successfully" &>/dev/null
            else
                slack "training $N_MODS replicates of distilled lentiMPRA models for $CELLTYPE completed successfully" &>/dev/null
            fi
        fi
	else
        if [ "$evoaug" = true ]; then
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled lentiMPRA models for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            else
		        slack "training $N_MODS replicates of distilled lentiMPRA models for $CELLTYPE w/ EvoAug exited with error code $exit_code"
            fi
        else
            if [ "$downsample" = true ]; then
                slack "training $N_MODS replicates of distilled downsampled lentiMPRA models for $CELLTYPE exited with error code $exit_code"
            else
                slack "training $N_MODS replicates of distilled lentiMPRA models for $CELLTYPE exited with error code $exit_code"
            fi
        fi
	fi
fi
exit "$exit_code"