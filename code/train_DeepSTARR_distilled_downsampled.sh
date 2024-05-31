# train an replicates of distilled DeepSTARR models from all downsampled ensembles using distilled training data

N_MODS=10
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
# MODELS_DIR=../results/DeepSTARR_lr-decay
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble # for wandb logger
EVOAUG=false

# if [ "$EVOAUG" = true ]; then
# 	MODELS_DIR=../results/DeepSTARR_evoaug
# fi
MODELS_DIR=../results/DeepSTARR_lr-decay/sanity_check
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    ENSEMBLE_DIR=${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
    OUTDIR=${ENSEMBLE_DIR}/ensemble_distilled
    mkdir -p $OUTDIR
    for i in $(seq 1 $N_MODS)
    do
        # echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        # ENSEMBLE_DIR=${MODELS_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}
        # # OUTDIR=${ENSEMBLE_DIR}/ensemble_distilled
        # OUTDIR=${ENSEMBLE_DIR}/sanity_check/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled
        # mkdir -p $OUTDIR
        if [ "$EVOAUG" = true ]; then 
            echo "python train_DeepSTARR.py --ix $i \
                                            --out $OUTDIR \
                                            --data $DATA \
                                            --plot \
                                            --config $CONFIG \
                                            --project $PROJECT_NAME \
                                            --lr_decay \
                                            --distill ${ENSEMBLE_DIR}/ensemble_avg_y_train.npy \
                                            --downsample ${DOWNSAMPLE_ARR[$p]} \
                                            --evoaug" 
        else
            echo "python train_DeepSTARR.py --ix $i \
                                            --out $OUTDIR \
                                            --data $DATA \
                                            --plot \
                                            --config $CONFIG \
                                            --project $PROJECT_NAME \
                                            --lr_decay \
                                            --distill ${ENSEMBLE_DIR}/ensemble_avg_y_train.npy \
                                            --downsample ${DOWNSAMPLE_ARR[$p]}" 
        fi
    done | simple_gpu_scheduler --gpus 2,3,4,5
done 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$EVOAUG" = true ]; then
		    slack "training ensemble distilled models (n=${N_MODS}) from ${MODELS_DIR} (downsampled) w/ EvoAug completed successfully" &>/dev/null
        else
            slack "training ensemble distilled models (n=${N_MODS}) from ${MODELS_DIR} (downsampled) completed successfully" &>/dev/null
        fi
	else
        if [ "$EVOAUG" = true ]; then
		    slack "training ensemble distilled models (n=${N_MODS}) from ${MODELS_DIR} (downsampled) w/ EvoAug exited with error code $exit_code"
        else
            slack "training ensemble distilled models (n=${N_MODS}) from ${MODELS_DIR} (downsampled) exited with error code $exit_code"
        fi
	fi
fi
exit "$exit_code"