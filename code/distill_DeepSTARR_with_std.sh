# trains a single distilled DeepSTARR model that predicts uncertainty (stdev) and mean

OUTDIR=../results/DeepSTARR_lr-decay/distilled_with_std
DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_distilled_with_std

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

CUDA_VISIBLE_DEVICES=4,5 python train_DeepSTARR.py --ix 1 --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --distill --predict_std

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training distilled DeepSTARR model with standard deviation prediction completed successfully" &>/dev/null
	else
		slack "training distilled DeepSTARR model with standard deviation prediction  exited with error code $exit_code"
	fi
fi
exit "$exit_code"