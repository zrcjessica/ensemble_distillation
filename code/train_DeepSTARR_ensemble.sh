ENSEMBLE_SIZE=10
OUTDIR=../results/DeepSTARR_base
DATA=../data/DeepSTARR/Sequences_activity_all.h5

mkdir -p $OUTDIR

for i in $(seq 1 $ENSEMBLE_SIZE)
do 
    echo "python train_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot"
done | simple_gpu_scheduler --gpus 4,5,6