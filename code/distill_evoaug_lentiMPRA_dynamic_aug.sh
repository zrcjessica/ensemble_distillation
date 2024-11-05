# train distilled ResidualBInd models w/ mean+aleatoric+epistemic using dynamic augmentations

AUG=evoaug # evoaug/mutagenesis/random 
APPEND=false
CELLTYPE=K562
ENSEMBLE_DIR=../results/lentiMPRA_aleatoric/${CELLTYPE}
DATA_DIR=../data/lentiMPRA
DATA=${DATA_DIR}/${CELLTYPE}_distillation_data_with_epistemic.h5
CONFIG=../config/lentiMPRA.yaml
PROJECT_NAME=ResidualBind_dynamic_aug
NMODS=10 # nr. of replicates to train
OUTDIR=../results/ResidualBind_dynamic_aug/${CELLTYPE}/${AUG}
if [ "$APPEND" = true ]; then
	OUTDIR=${OUTDIR}/append
fi 
mkdir -p $OUTDIR 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for i in $(seq 4 $NMODS)
do
	if [ "$APPEND" = true ]; then
		echo "python train_dynamic_aug_lentiMPRA.py --ix $i \
				--out $OUTDIR \
				--data $DATA \
				--plot \
				--config $CONFIG \
				--project $PROJECT_NAME \
				--lr_decay \
				--aug $AUG \
				--celltype $CELLTYPE \
				--ensemble_dir $ENSEMBLE_DIR \
				--append"
	else 
		echo "python train_dynamic_aug_lentiMPRA.py --ix $i \
				--out $OUTDIR \
				--data $DATA \
				--plot \
				--config $CONFIG \
				--project $PROJECT_NAME \
				--lr_decay \
				--aug $AUG \
				--celltype $CELLTYPE \
				--ensemble_dir $ENSEMBLE_DIR"
	fi
done | simple_gpu_scheduler --gpus 4,5,6,7


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$APPEND" = true ]; then
			slack "training distilled mean+aleatoric+epistemic ResidualBind models w/ dynamic augs ($AUG, appended) completed successfully" &>/dev/null
		else
			slack "training distilled mean+aleatoric+epistemic ResidualBind models w/ dynamic augs ($AUG) completed successfully" &>/dev/null
		fi 
	else
		if [ "$APPEND" = true ]; then
			slack "training distilled mean+aleatoric+epistemic ResidualBind models w/ dynamic augs ($AUG, appended) exited with error code $exit_code"
		else 
			slack "training distilled mean+aleatoric+epistemic ResidualBind models w/ dynamic augs ($AUG) exited with error code $exit_code"
		fi
	fi
fi
exit "$exit_code"