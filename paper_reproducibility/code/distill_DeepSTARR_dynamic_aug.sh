# train distilled DeepSTARR models w/ mean+std using dynamic augmentations

AUG='random' # evoaug/mutagenesis/random 
APPEND=false # if true, augmented seqs are appended to each mini batch
ENSEMBLE_DIR=../results/DeepSTARR_ensemble_NEW/ # path to pre-trained ensemble of models 
DATA_DIR=../data/DeepSTARR_ensemble_NEW # path to training data directory
DATA=${DATA_DIR}/all_data_with_ensemble_metrics_hierarchical.h5 # path to training data 
CONFIG=../config/DeepSTARR.yaml # path to model config
PROJECT_NAME=DeepSTARR_dynamic_aug # project name for WandB logging 
NMODS=10 # nr. of distilled models to train
OUTDIR=../results/DeepSTARR_dynamic_aug/${AUG} # path to output directory

if [ "$APPEND" = true ]; thene
	OUTDIR=${OUTDIR}/append
fi 
mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for i in $(seq 1 $NMODS)
do
	if [ "$APPEND" = true ]; then
		echo "python train_dynamic_aug_DeepSTARR.py --ix $i \
				--out $OUTDIR \
				--data $DATA \
				--plot \
				--config $CONFIG \
				--project $PROJECT_NAME \
				--lr_decay \
				--aug $AUG \
				--ensemble_dir $ENSEMBLE_DIR \
				--append"
	else 
		echo "python train_dynamic_aug_DeepSTARR.py --ix $i \
				--out $OUTDIR \
				--data $DATA \
				--plot \
				--config $CONFIG \
				--project $PROJECT_NAME \
				--lr_decay \
				--aug $AUG \
				--ensemble_dir $ENSEMBLE_DIR"
	fi
done | simple_gpu_scheduler --gpus 0,3,4,5,6


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$APPEND" = true ]; then
			slack "training distilled mean+std DeepSTARR models w/ dynamic augs ($AUG, appended) completed successfully" &>/dev/null
		else
			slack "training distilled mean+std DeepSTARR models w/ dynamic augs ($AUG) completed successfully" &>/dev/null
		fi 
	else
		if [ "$APPEND" = true ]; then
			slack "training distilled mean+std DeepSTARR models w/ dynamic augs ($AUG, appended) exited with error code $exit_code"
		else 
			slack "training distilled mean+std DeepSTARR models w/ dynamic augs ($AUG, appended) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"