# trains distilled DeepSTARR models that predict epistemic uncertainty (stdev) and mean

OUTDIR=../results/DeepSTARR_ensemble_NEW # path to output directory
# OUTDIR=../results/DeepSTARR_lr-decay
# DATA_DIR=../data/DeepSTARR
DATA_DIR=../data/DeepSTARR_ensemble_NEW # path to training data 
CONFIG=../config/DeepSTARR.yaml # path to model config 
PROJECT_NAME=DeepSTARR_distilled_epistemic # name for WandB project logging 
NMODS=10 # number of models to train 

### boolean vars (toggle true/false)
# train w/ evoaug
evoaug=true
if [ "$evoaug" = true ]; then
    OUTDIR=../results/DeepSTARR_evoaug_NEW
	DATA_DIR=../data/DeepSTARR_evoaug 
fi
downsample=true # train on subset of training data

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
	# train downsampled models
	echo "training downsampled models"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		DOWNSAMPLE_OUTDIR=${OUTDIR}/downsample_${DOWNSAMPLE_ARR[$p]}/distilled_with_std
		DOWNSAMPLE_DATA=${DATA_DIR}/downsample${DOWNSAMPLE_ARR[$p]}_all_data_with_ensemble_metrics_hierarchical.h5
		mkdir -p $DOWNSAMPLE_OUTDIR
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
		for i in $(seq 1 $NMODS)
		do
			if [ "$evoaug" = true ]; then
				echo "python train_stdev_DeepSTARR.py --ix $i --out $DOWNSAMPLE_OUTDIR --data $DOWNSAMPLE_DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug --downsample ${DOWNSAMPLE_ARR[$p]}"
			else
				echo "python train_stdev_DeepSTARR.py --ix $i --out $DOWNSAMPLE_OUTDIR --data $DOWNSAMPLE_DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --downsample ${DOWNSAMPLE_ARR[$p]}"
			fi
		done | simple_gpu_scheduler --gpus 0,1,3,5,6
	done 
else 
	# train multiple replicates
	DATA=${DATA_DIR}/all_data_with_ensemble_metrics_hierarchical.h5
	OUTDIR=$OUTDIR/distilled_with_std 
	mkdir -p $OUTDIR
	for i in $(seq 1 $NMODS)
	do 
		if [ "$evoaug" = true ]; then
			echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
		else
			echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
		fi
	done | simple_gpu_scheduler --gpus 0,1,3,5,6
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$evoaug" = true ]; then
			if [ "$downsample" = true ]; then 
				slack "training downsampled distilled DeepSTARR models (+EvoAug) with standard deviation prediction completed successfully" &>/dev/null
			else 
				slack "training distilled DeepSTARR models (+EvoAug) with standard deviation prediction completed successfully" &>/dev/null
			fi
		else
			if [ "$downsample" = true ]; then 
				slack "training downsampled distilled DeepSTARR models with standard deviation prediction completed successfully" &>/dev/null
			else 
				slack "training distilled DeepSTARR models with standard deviation prediction completed successfully" &>/dev/null
			fi
		fi
	else
		if [ "$evoaug" = true ]; then
			if [ "$downsample" = true ]; then 
				slack "training downsampled distilled DeepSTARR models (+EvoAug) with standard deviation prediction  exited with error code $exit_code"
			else 
				slack "training distilled DeepSTARR models (+EvoAug) with standard deviation prediction  exited with error code $exit_code"
			fi
		else
			if [ "$downsample" = true ]; then  
				slack "training downsampled distilled DeepSTARR models with standard deviation prediction  exited with error code $exit_code"
			else 
				slack "training distilled DeepSTARR models with standard deviation prediction  exited with error code $exit_code"
			fi
		fi
	fi
fi
exit "$exit_code"