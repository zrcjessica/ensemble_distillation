METHOD=shap
ATTR_FILES_DIR=../results/DeepSTARR_lr-decay/distilled_with_std
REF_FILE=../results/DeepSTARR_lr-decay/average_top500_shap.npy

python analyze_attr_scores.py --files_dir $ATTR_FILES_DIR --reference $REF_FILE --method $METHOD --rmse 


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "analyzing $METHOD scores for distilled DeepSTARR model with standard deviation prediction completed successfully" &>/dev/null
	else
		slack "analyzing $METHOD scores for distilled DeepSTARR model with standard deviation prediction  exited with error code $exit_code"
	fi
fi
exit "$exit_code"