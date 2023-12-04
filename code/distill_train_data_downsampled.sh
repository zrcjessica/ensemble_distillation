# runs get_train_data_for_distillation.py
# for al DeepSTARR ensembles trained on downsampled training data

MODEL_DIR=../results/DeepSTARR_lr-decay
N_MODS=10
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
DATA=../data/DeepSTARR/Sequences_activity_all.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        echo "python get_train_data_for_distillation.py --model_dir ${MODEL_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --n_mods $N_MODS --data $DATA "
    done 
done | simple_gpu_scheduler --gpus 4,5,6

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "running get_train_data_for_distillation.py for DeepSTARR_lr-decay downsampled models completed successfully" &>/dev/null
	else
		slack "running get_train_data_for_distillation.py for DeepSTARR_lr-decay downsampled models exited with error code $exit_code"
	fi
fi
exit "$exit_code"