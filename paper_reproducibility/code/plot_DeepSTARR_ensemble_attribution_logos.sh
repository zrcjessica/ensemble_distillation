# runs plot_ensemble_saliency_logos.py for DeepSTARR ensembles

# SALIENCY_DIR=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay
SALIENCY_DIR=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay/ensemble_distilled
METHOD=shap
AVG_FILE=$HOME/ensemble_distillation/results/DeepSTARR_lr-decay/average_top500_${METHOD}.npy

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# python plot_ensemble_saliency_logos.py --saliency_dir $SALIENCY_DIR --average $AVG_FILE
python plot_ensemble_attribution_logos.py --files_dir $SALIENCY_DIR --average $AVG_FILE --method $METHOD

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "Plotting $METHOD logos for top 10 seqs from DeepSTARR_lr-decay ensemble (distilled) completed successfully" &>/dev/null
	else
		slack "Plotting $METHOD logos for top 10 seqs from DeepSTARR_lr-decay ensemble (distilled) exited with error code $exit_code"
	fi
fi
exit "$exit_code"