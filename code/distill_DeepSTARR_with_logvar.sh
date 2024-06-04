# trains a single distilled DeepSTARR model that predicts uncertainty (stdev) and mean

OUTDIR=../results/DeepSTARR_lr-decay/distilled_with_logvar
DATA_DIR=../data/DeepSTARR
DATA=${DATA_DIR}/all_data_with_ensemble_metrics_hierarchical.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_distilled_with_logvar
NMODS=10

# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
	OUTDIR=../results/DeepSTARR_evoaug/distilled_with_logvar
	DATA=${DATA_DIR}/evoaug/all_data_with_ensemble_metrics_hierarchical.h5
fi

mkdir -p $OUTDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# train multiple replicates
for i in $(seq 1 $NMODS)
do 
	if [ "$evoaug" = true ]; then
		echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --logvar"
	else
		echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --logvar"
	fi
done | simple_gpu_scheduler --gpus 1,2,3,4,5,6,7

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$evoaug" = true ]; then
			slack "training distilled DeepSTARR models (+EvoAug) with log variance prediction completed successfully" &>/dev/null
		else
			slack "training distilled DeepSTARR models with log variance prediction completed successfully" &>/dev/null
		fi
	else
		if [ "$evoaug" = true ]; then
			slack "training distilled DeepSTARR models (+EvoAug) with log variance prediction  exited with error code $exit_code"
		else 
			slack "training distilled DeepSTARR models with log variance prediction  exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"