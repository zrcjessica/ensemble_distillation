# analyze attribution scores (shap/saliency) for a set of lentiMPRA models 
# can set DOWNSAMPLE and DISTILLED boolean variables  

### boolean vars 
# DOWNSAMPLED=false # toggle true/false to analyze models trained on downsampled data
DISTILLED=true # toggle true/false to analyze distilled models 
# EPISTEMIC=true # if set, analyze distilled models w/ epistemic output; only used if DISTILLED=true
EVOAUG=false

### define method (saliency/shap)
METHOD=saliency 

### which celltype celltype
CELLTYPE=HepG2

### top n seqs
TOP_N=1000

### define root directory where all files are located 
FILES_DIR=../results/lentiMPRA/$CELLTYPE

### define path to reference file 
REF_FILE=${FILES_DIR}/avg_top1000_${CELLTYPE}-mean_${METHOD}.npy

if [ "$DISTILLED" = true ]; then
    FILES_DIR=../results/distilled_lentiMPRA_epistemic/$CELLTYPE 
fi 

python analyze_lentiMPRA_attr_scores.py \
            --files_dir ${FILES_DIR} \
            --reference ${REF_FILE} \
            --method $METHOD \
            --top_n $TOP_N \
            --celltype $CELLTYPE \
            --rmse \
            --var

# message the user on slack if possible
exit_code="$?"
if command -v 'slack' &>/dev/null; then
    if [ "$exit_code" -eq 0 ]; then
        if [ "$DISTILLED" = true ]; then
            slack "running analyze_lentiMPRA_attr_scores.py on distilled lentiMPRA $CELLTYPE $METHOD scores completed successfully" &>/dev/null
        else 
            slack "running analyze_lentiMPRA_attr_scores.py on lentiMPRA $CELLTYPE $METHOD scores completed successfully" &>/dev/null
        fi
    else 
        if [ "$DISTILLED" = true ]; then
            slack "running analyze_lentiMPRA_attr_scores.py on distilled lentiMPRA $CELLTYPE $METHOD scores exited with error code $exit_code" 
        else 
            slack "running analyze_lentiMPRA_attr_scores.py on lentiMPRA $CELLTYPE $METHOD scores exited with error code $exit_code" 
        fi 
    fi
fi
exit "$exit_code"