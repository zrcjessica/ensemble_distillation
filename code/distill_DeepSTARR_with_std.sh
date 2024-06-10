# trains a single distilled DeepSTARR model that predicts uncertainty (stdev) and mean

OUTDIR=../results/DeepSTARR_lr-decay/distilled_with_std_REDO
# OUTDIR=../results/DeepSTARR_lr-decay
DATA_DIR=../data/DeepSTARR
DATA=${DATA_DIR}/all_data_with_ensemble_metrics_hierarchical.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_distilled_with_std_REDO
NMODS=10

### boolean vars (toggle true/false)
# train w/ evoaug
evoaug=false
if [ "$evoaug" = true ]; then
    OUTDIR=../results/DeepSTARR_evoaug/distilled_with_std
	# OUTDIR=../results/DeepSTARR_evoaug
	DATA=${DATA_DIR}/evoaug/all_data_with_ensemble_metrics_hierarchical.h5
fi
downsample=true 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [ "$downsample" = true ]; then
	# train downsampled models
	echo "training downsampled models"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do	
		DOWNSAMPLE_OUTDIR=${OUTDIR}/downsample_${DOWNSAMPLE_ARR[$p]}
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
		done | simple_gpu_scheduler --gpus 0,1,2,3,4
	done 
else 
	# train multiple replicates
	OUTDIR=$OUTDIR/distilled_with_std 
	mkdir -p $OUTDIR
	for i in $(seq 1 $NMODS)
	do 
		if [ "$evoaug" = true ]; then
			echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay --evoaug"
		else
			echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --plot --config $CONFIG --project $PROJECT_NAME --lr_decay"
		fi
	done | simple_gpu_scheduler --gpus 0,1,2,3,4
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