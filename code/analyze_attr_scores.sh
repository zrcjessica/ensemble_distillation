# analyzes attribution scores (shap/saliency) for a set of DeepSTARR models 
# can set DOWNSAMPLE and DISTILLED boolean variables  

DOWNSAMPLED=true # toggle true/false
DISTILLED=true # toggle true/false
METHOD=saliency # set saliency/shap
FILES_DIR=../results/DeepSTARR_lr-decay

if [ "$DOWNSAMPLED" = true ]; then
	echo "Analyzing $METHOD scores for DeepSTARR models trained on subset of training data"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        if [ "$DISTILLED" = true ]; then
            python analyze_attr_scores.py --files_dir ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled --reference ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/average_top500_saliency.npy --method $METHOD --rmse --var
        else 
            python analyze_attr_scores.py --files_dir ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} --method $METHOD --rmse --var
        fi
    done 
else
    if [ "$DISTILLED" = true ]; then 
        python analyze_attr_scores.py --files_dir $FILES_DIR/ensemble_distilled --reference ${FILES_DIR}/average_top500_saliency.npy --method $METHOD --rmse --var
    else
        python analyze_attr_scores.py --files_dir $FILES_DIR --method $METHOD --rmse --var
    fi
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (downsampled & distilled) $METHOD scores completed successfully" &>/dev/null 
            else 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (downsampled & not distilled) $METHOD scores completed successfully" &>/dev/null
            fi 
        else
            if ["$DISTILLED" = true ]; then 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (trained on full dataset & distilled) $METHOD scores completed successfully" &>/dev/null 
            else 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (trained on full dataset & not distilled) $METHOD scores completed successfully" &>/dev/null 
            fi 
        fi 
    else 
        if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (downsampled & distilled) $METHOD scores exited with error code $exit_code" 
            else 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (downsampled & not distilled) $METHOD scores exited with error code $exit_code"
            fi 
        else
            if ["$DISTILLED" = true ]; then 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (trained on full dataset & distilled) $METHOD scores exited with error code $exit_code"
            else 
                slack "running analyze_attr_scores.py on DeepSTARR_lr-decay (trained on full dataset & not distilled) $METHOD scores exited with error code $exit_code"
            fi 
        fi 
    fi
fi
exit "$exit_code"