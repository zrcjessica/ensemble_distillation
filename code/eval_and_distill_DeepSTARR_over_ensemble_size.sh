# run ensemble_predict_DeepSTARR_over_ensemble_size.py 
# examines different sizes of DeepSTARR ensembles, up to 25

MODEL_DIR=../results/DeepSTARR_ensemble_size # path to DeepSTARR models 
DATA=../data/DeepSTARR/Sequences_activity_all.h5 # path to STARR-seq data 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR_over_ensemble_size.py --model_dir $MODEL_DIR --n_mods 5 10 15 20 25 --data $DATA --out 
CUDA_VISIBLE_DEVICES=7 python ensemble_predict_DeepSTARR_over_ensemble_size.py --model_dir $MODEL_DIR --n_mods 2 3 4 --data $DATA

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		# slack "running ensemble_predict_DeepSTARR_over_ensemble_size.py for 5, 10, 15, 20, 25 models completed successfully" &>/dev/null 
		slack "running ensemble_predict_DeepSTARR_over_ensemble_size.py for 2,3,4 models completed successfully" &>/dev/null 
	else 
		# slack "running ensemble_predict_DeepSTARR_over_ensemble_size.py for 5, 10, 15, 20, 25 models exited with error code $exit_code"
		slack "running ensemble_predict_DeepSTARR_over_ensemble_size.py for 2,3,4 models exited with error code $exit_code"
	fi 
fi
exit "$exit_code"
