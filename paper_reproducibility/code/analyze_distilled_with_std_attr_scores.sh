# analyze attr scores for quantile binned DeepSTARR test sequences from distilled models with stdev prediction
# reference scores are the average attr scores obtained from the ensemble average of models (no stdev prediction)

QUANTILES_ARR=( 0.25 0.5 0.75 1.0 )
REF_DIR=../results/DeepSTARR_lr-decay #path to dir containing attribution scores for reference values
MODS_DIR=../results/DeepSTARR_lr-decay/distilled_with_std #path to dir containing attr scores for models to evaluate
METHOD=shap #shap/saliency

for q in "${!QUANTILES_ARR[@]}"
do
    echo "quantile $q"
    python analyze_binned_attr_scores.py --files_dir $MODS_DIR --reference ${REF_DIR}/average_q${q}_100seqs_${METHOD}.npy --quantile $q --method $METHOD --rmse 
done

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "analyzing $METHOD scores for distilled DeepSTARR models (with std. prediction) for binned sequences completed successfully" &>/dev/null
	else
		slack "analyzing $METHOD scores for distilled DeepSTARR models (with std. prediction) for binned sequences exited with error code $exit_code"
	fi
fi
exit "$exit_code"