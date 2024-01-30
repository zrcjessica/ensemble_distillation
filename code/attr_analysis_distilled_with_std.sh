# run attribution analysis on distilled DeepSTARR models trained to predict mean+std 
# for test sequences selected from quantile binning y_test for Developmental enhancers

METHOD=shap # set saliency or shap
N_MODS=10
MODELS_DIR=../results/DeepSTARR_lr-decay/distilled_with_std
#MODELS_DIR=../results/DeepSTARR_lr-decay # calculate ensemble average of attribution maps
SEQS_FILE=../results/DeepSTARR_lr-decay/distilled_with_std/100_binned_seqs.csv
DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# mkdir -p ${MODELS_DIR}/${METHOD}

CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_attr_analysis_for_binned_seqs.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --std --predict
#CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_attr_analysis_for_binned_seqs.py --model_dir $MODELS_DIR --seqs $SEQS_FILE --n_mods $N_MODS --data $DATA --method $METHOD --average 

# # message the user on slack if possible
# exit_code="$?"
# if command -v 'slack' &>/dev/null; then
#     if [ "$exit_code" -eq 0 ]; then
# 		slack "performing $METHOD analysis on distilled DeepSTARR model (mean+std) completed successfully" &>/dev/null
# 	else
# 		slack "performing $METHOD analysis on distilled DeepSTARR model (mean+std) exited with error code $exit_code"
# 	fi
# fi
# exit "$exit_code"

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "performing $METHOD analysis for distilled DeepSTARR models (w/ std. prediction) for binned sequences completed successfully" &>/dev/null
	else
		slack "performing $METHOD analysis for distilled DeepSTARR models (w/ std. prediction) for binned sequences exited with error code $exit_code"
	fi
fi
exit "$exit_code"
