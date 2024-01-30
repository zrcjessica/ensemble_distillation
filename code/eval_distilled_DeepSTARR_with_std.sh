# evaluates the ensemble performance of distilled DeepSTARR models trained to predict stdev + mean
# generates plot of true vs. predicted values 

MODEL_DIR=../results/DeepSTARR_lr-decay/distilled_with_std
N_MODS=10
DATA=../data/DeepSTARR/all_data_with_ensemble_metrics_hierarchical.h5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

CUDA_VISIBLE_DEVICES=4 python ensemble_predict_DeepSTARR.py --model_dir $MODEL_DIR --n_mods $N_MODS --data $DATA --eval --std --plot


# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "evaluating performance of distilled DeepSTARR models with standard deviation prediction completed successfully" &>/dev/null
	else
		slack "evaluating performance of distilled DeepSTARR models with standard deviation prediction  exited with error code $exit_code"
	fi
fi
exit "$exit_code"
