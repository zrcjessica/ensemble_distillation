# train distilled DeepSTARR models w/ mean+std using dynamic augmentations

AUG='random' # evoaug/mutagenesis/random 
APPEND=false 
ENSEMBLE_DIR=../results/DeepSTARR_lr-decay/
DATA_DIR=../data/DeepSTARR
DATA=${DATA_DIR}/all_data_with_ensemble_metrics_hierarchical.h5
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_dynamic_aug
NMODS=10 # nr. of replicates to train
OUTDIR=../results/DeepSTARR_dynamic_aug/${AUG}

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
done | simple_gpu_scheduler --gpus 0,2,3,4,5


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