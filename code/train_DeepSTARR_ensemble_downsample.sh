ENSEMBLE_SIZE=10
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
    OUTDIR=../results/DeepSTARR_base/downsample_${DOWNSAMPLE_ARR[$p]}
    mkdir -p $OUTDIR
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --downsample ${DOWNSAMPLE_ARR[$p]} --config $CONFIG --project $PROJECT_NAME"
    done 
done | simple_gpu_scheduler --gpus 4,5,6
