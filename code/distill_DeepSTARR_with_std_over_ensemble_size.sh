# trains distilled DeepSTARR models that predict epistemic uncertainty (stdev) and mean

DATA_DIR=../data/DeepSTARR_ensemble_over_size
CONFIG=../config/DeepSTARR.yaml
PROJECT_NAME=DeepSTARR_ensemble_size
# N_ARR=( 2 3 4 5 10 15 20 25 )
N_ARR=( 4 5 10 15 20 25 )
NMODS=10


### boolean vars (toggle true/false)
# # train w/ evoaug
# evoaug=false
# if [ "$evoaug" = true ]; then
#     OUTDIR=../results/DeepSTARR_evoaug_NEW
# 	DATA_DIR=../data/DeepSTARR_evoaug 
# fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for n in "${!N_ARR[@]}"
do
	echo "training distilled models based on ensemble of ${N_ARR[$n]} DeepSTARR models"
	OUTDIR=../results/DeepSTARR_ensemble_size/${N_ARR[$n]}mod_distilled_with_std
	mkdir -p $OUTDIR 
	DATA=${DATA_DIR}/DeepSTARR_${N_ARR[$n]}mod_distillation_data_hierarchical.h5
	for i in $(seq 1 $NMODS)
	do
		echo "python train_stdev_DeepSTARR.py --ix $i --out $OUTDIR --data $DATA --config $CONFIG --project $PROJECT_NAME --lr_decay --nmods ${N_ARR[$n]}"
	done | simple_gpu_scheduler --gpus 0,1,2,3,4,5
	slack "finishing training 10 distilled DeepSTARR models from ensemble of ${N_ARR[$n]} models"
done 

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "training distilled DeepSTARR models based on ensembles of size n=2,3,4,5,10,15,20,25 completed successfully" &>/dev/null 
	else 		
		slack "training distilled DeepSTARR models based on ensembles of size n=2,3,4,5,10,15,20,25 exited with error code $exit_code" 
	fi
fi 
exit "$exit_code"
