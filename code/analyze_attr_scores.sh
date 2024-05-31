# analyze attribution scores (shap/saliency) for a set of DeepSTARR models 
# can set DOWNSAMPLE and DISTILLED boolean variables  

### boolean vars 
DOWNSAMPLED=false # toggle true/false
DISTILLED=true # toggle true/false
STD=true # only used if DISTILLED=true
EVOAUG=false

### define method (saliency/shap)
METHOD=shap 

### which enhancer output 
ENHANCER=Dev 

### top n seqs
TOP_N=1000

### define root directory where all files are located 
FILES_DIR=../results/DeepSTARR_lr-decay


if [ "$DOWNSAMPLED" = true ]; then
	echo "Analyzing $METHOD scores for DeepSTARR models trained on subset of training data"
	DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
	for p in "${!DOWNSAMPLE_ARR[@]}"
	do
		echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
        if [ "$DISTILLED" = true ]; then
            if [ "$STD" = true ]; then 
                python analyze_attr_scores.py \
                --files_dir ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/distilled_with_std \
                --reference ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/avg_top${TOP_N}_${ENHANCER}_saliency.npy \
                --method $METHOD \
                --top_n $TOP_N \
                --enhancer $ENHANCER \
                --rmse \
                --var
            else  
                python analyze_attr_scores.py \
                --files_dir ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/ensemble_distilled \
                --reference ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]}/avg_top${TOP_N}_${ENHANCER}_saliency.npy \
                --method $METHOD \
                --top_n $TOP_N \
                --enhancer $ENHANCER \
                --rmse \
                --var  
            fi
        else 
            python analyze_attr_scores.py \ 
            --files_dir ${FILES_DIR}/downsample_${DOWNSAMPLE_ARR[$p]} \
            --method $METHOD \
            --top_n $TOP_N \
            --enhancer $ENHANCER \
            --rmse \
            --var
        fi
    done 
else
    if [ "$DISTILLED" = true ]; then
        if [ "$STD" = true ]; then  
            python analyze_attr_scores.py \
            --files_dir $FILES_DIR/distilled_with_std \
            --reference ${FILES_DIR}/avg_top${TOP_N}_${ENHANCER}_saliency.npy \
            --method $METHOD \
            --top_n $TOP_N \
            --enhancer $ENHANCER \
            --rmse \
            --var
        else 
            python analyze_attr_scores.py \
            --files_dir $FILES_DIR/ensemble_distilled \
            --reference ${FILES_DIR}/avg_top${TOP_N}_${ENHANCER}_saliency.npy \
            --method $METHOD \
            --top_n $TOP_N \
            --enhancer $ENHANCER \
            --rmse \
            --var
        fi 
    else
        python analyze_attr_scores.py --files_dir $FILES_DIR --method $METHOD --top_n $TOP_N --enhancer $ENHANCER --rmse --var 
    fi
fi

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                if [ "$STD" = true ]; then 
                    slack "running analyze_attr_scores.py on $FILES_DIR (downsampled, distilled w/ std) $METHOD scores completed successfully" &>/dev/null 
                else 
                    slack "running analyze_attr_scores.py on $FILES_DIR (downsampled & distilled) $METHOD scores completed successfully" &>/dev/null 
                fi
            else 
                slack "running analyze_attr_scores.py on $FILES_DIR (downsampled) $METHOD scores completed successfully" &>/dev/null
            fi 
        else
            if [ "$DISTILLED" = true ]; then 
                if [ "$STD" = true ]; then 
                    slack "running analyze_attr_scores.py on $FILES_DIR (distilled w/ std) $METHOD scores completed successfully" &>/dev/null 
                else 
                    slack "running analyze_attr_scores.py on $FILES_DIR (distilled) $METHOD scores completed successfully" &>/dev/null 
                fi
            else 
                slack "running analyze_attr_scores.py on $FILES_DIR $METHOD scores completed successfully" &>/dev/null 
            fi 
        fi 
    else 
        if [ "$DOWNSAMPLED" = true ]; then
            if [ "$DISTILLED" = true ]; then
                if [ "$STD" = true ]; then 
                    slack "running analyze_attr_scores.py on $FILES_DIR (downsampled, distilled w/ std) $METHOD scores exited with error code $exit_code" 
                else 
                    slack "running analyze_attr_scores.py on $FILES_DIR (downsampled & distilled) $METHOD scores exited with error code $exit_code" 
                fi
            else 
                slack "running analyze_attr_scores.py on $FILES_DIR (downsampled) $METHOD scores exited with error code $exit_code"
            fi 
        else
            if [ "$DISTILLED" = true ]; then 
                if [ "$STD" = true ]; then
                    slack "running analyze_attr_scores.py on $FILES_DIR (distilled w/ std) $METHOD scores exited with error code $exit_code"
                else 
                    slack "running analyze_attr_scores.py on $FILES_DIR (distilled) $METHOD scores exited with error code $exit_code"
                fi
            else 
                slack "running analyze_attr_scores.py on $FILES_DIR $METHOD scores exited with error code $exit_code"
            fi 
        fi 
    fi
fi
exit "$exit_code"