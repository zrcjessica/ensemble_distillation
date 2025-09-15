# train an ensemble of DeepSTARR models on downsampled training data
# loops through an array of different downsampling proportions

ENSEMBLE_SIZE=10
DOWNSAMPLE_ARR=( 0.1 0.25 0.5 0.75 )
DATA=../data/DeepSTARR/Sequences_activity_all.h5
OUTDIR=../results/DeepSTARR_lr-decay
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

for p in "${!DOWNSAMPLE_ARR[@]}"
do
    echo "downsample p = ${DOWNSAMPLE_ARR[$p]}"
    for i in $(seq 1 $ENSEMBLE_SIZE)
    do 
        python test_random_downsample.py --ix $i --out $OUTDIR --data $DATA --downsample ${DOWNSAMPLE_ARR[$p]}
    done 
done

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
		slack "checking random sampling for downsampling completed successfully" &>/dev/null
	else
		slack "checking random sampling for downsampling exited with error code $exit_code"
	fi
fi
exit "$exit_code"