# run attribution analysis on distilled DeepSTARR models trained to predict mean+std 

METHOD=shap # set saliency or shap
STD=true # true/false
N_MODS=1
MODELS_DIR=../results/DeepSTARR_lr-decay/distilled_with_std
DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

CUDA_VISIBLE_DEVICES=4,5,6 python DeepSTARR_ensemble_attr_analysis.py --model_dir $MODELS_DIR --n_mods $N_MODS --data $DATA --method $METHOD --std --ref_size 1000

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "performing $METHOD analysis on distilled DeepSTARR model (mean+std) completed successfully" &>/dev/null
	else
		slack "performing $METHOD analysis on distilled DeepSTARR model (mean+std) exited with error code $exit_code"
	fi
fi
exit "$exit_code"