
OUTDIR=../results/DeepSTARR_base_TEST
DATA=../data/DeepSTARR/Sequences_activity_all.h5

mkdir -p $OUTDIR

python train_DeepSTARR.py --ix 1 --out $OUTDIR --data $DATA --plot

