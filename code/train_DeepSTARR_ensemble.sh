ENSEMBLE_SIZE=10
OUTDIR=../results/DeepSTARR_lr-decay
DATA=../data/DeepSTARR/Sequences_activity_all.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for i in $(seq 1 $ENSEMBLE_SIZE)
do 
    echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
done | simple_gpu_scheduler --gpus 4,5,6,7
