
OUTDIR=../results/DeepSTARR_base_TEST
DATA=../data/DeepSTARR/Sequences_activity_all.h5

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=1 python train_DeepSTARR.py --ix 1 --out $OUTDIR --data $DATA --plot

