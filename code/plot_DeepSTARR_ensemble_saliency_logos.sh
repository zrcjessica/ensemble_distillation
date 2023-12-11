# runs plot_ensemble_saliency_logos.py for DeepSTARR ensembles

# SALIENCY_DIR=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay
SALIENCY_DIR=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay/ensemble_distilled
AVG_FILE=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay/average_top500_saliency.npy

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python plot_ensemble_saliency_logos.py --saliency_dir $SALIENCY_DIR --average $AVG_FILE

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "Plotting saliency logos for top 10 seqs from DeepSTARR_lr-decay distilled ensemble (full) completed successfully" &>/dev/null
	else
		slack "Plotting saliency logos for top 10 seqs from DeepSTARR_lr-decay distilled ensemble (full) exited with error code $exit_code"
	fi
fi
exit "$exit_code"