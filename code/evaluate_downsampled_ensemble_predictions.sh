# runs ensemble_predict_DeepSTARR.py for downsampled DeepSTARR ensembles

N_MODS=10
DATA=../data/DeepSTARR/Sequences_activity_all.h5
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
    MODEL_DIR=../results/DeepSTARR_lr-decay/downsample_${DOWNSAMPLE_ARR[$p]}
    echo "python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA"
done | simple_gpu_scheduler --gpus 4,5,6

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (downsampled, trained with LR decay) completed successfully" &>/dev/null
	else
		slack "running ensemble_predict_DeepSTARR.py on DeepSTARR_lr-decay (downsampled, trained with LR decay) exited with error code $exit_code"
	fi
fi
exit "$exit_code"