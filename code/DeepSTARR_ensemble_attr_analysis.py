import argparse
import utils # shap is imported by utils, must be imported before keras
import keras
from keras.models import load_model
import gc
import os
import numpy as np
import h5py
import pandas as pd
from tensorflow import Variable
import pandas as pd
from os.path import join

"""
Run Attribution Analysis on an ensemble of DeepSTARR models

This script performs attribution analysis (either saliency or DeepSHAP) on the top
predictions from a test set for an ensemble of DeepSTARR models 
(or multiple distilled DeepSTARR models). It can analyze predictions
for a specific promoter class and calculate either individual or average attribution scores
across the models in the ensemble.

Supported attribution methods:
- `saliency`: Computes gradients to highlight important input features.
- `shap`: Uses SHAP (SHapley Additive exPlanations) with options for dinucleotide shuffling as reference.

The script also allows optional adjustments for models trained with EvoAug and for averaging 
attribution scores across the ensemble.

Usage:
    python DeepSTARR_ensemble_attr_analysis.py --model_dir <path_to_model_files> --n_mods <number_of_models> --data <path_to_data_file> --out <output_directory> [additional options]

Arguments:
- `--model_dir`: Path to the directory storing trained model files.
- `--n_mods`: Number of models in the ensemble to evaluate.
- `--data`: Path to an HDF5 file containing train/validation/test data.
- `--out`: Directory where the results will be saved.
- `--top_n`: Number of top predictions from the test set to analyze (default: 500).
- `--enhancer`: Specifies the promoter class for sorting top predictions ('Dev' or 'Hk', default: 'Dev').
- `--head`: Specifies which output head to analyze ('mean', 'std', or 'logvar', default: 'mean').
- `--average`: Flag to calculate average attribution map across all models in the ensemble.
- `--method`: Attribution method to use ('saliency' or 'shap').
- `--dinuc_shuffle`: If using SHAP, set this to use dinucleotide-shuffled sequences as references.
- `--ref_size`: If using SHAP, specifies the size of the reference set (default: 100).
- `--evoaug`: Set this flag if using models trained with EvoAug.
- `--config`: Configuration file path (required if `--evoaug` is set).
- `--std`: Flag indicating if the model predicts standard deviation (i.e., a distilled model).

"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help='path directory storing trained ensemble of models')
    parser.add_argument("--n_mods", type=int, 
                        help="number of models to evaluate")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--top_n", type=int, default=500,
                        help='how many of top predictions from test set to analyze')
    parser.add_argument('--enhancer', type=str, default='Dev',
                        help='which class of predictions to sort top predictions for')
    parser.add_argument("--head", type=str, default='mean',
                        help='one of mean/std/logvar; determines which head to perform attribution analysis for')
    parser.add_argument("--average", action='store_true',
                        help='if set, calculate average saliency map across all models provided')
    parser.add_argument("--method", type=str,
                        help='saliency or shap; determines what method to use for attribution map')
    parser.add_argument("--dinuc_shuffle", action='store_true',
                        help='set if method=shap; if true, use dinucleotide shuffled seqs for shap reference per sequence')
    parser.add_argument("--ref_size", default=100, type=int,
                        help='set if method=shap; size of reference set')
    parser.add_argument("--evoaug", action='store_true',
                        help='set when using models trained w/ EvoAug')
    parser.add_argument("--config", default=None,
                        help='provide if --evoaug flag set; needed to load model from weights')
    parser.add_argument("--std", action='store_true',
                        help='if true, model predicts standard deviation (i.e. is a distilled model)')

    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data, std=args.std)

    # set output directory
    outdir = args.out
    if outdir is None:
        outdir=args.model_dir

    # sort y_test from largest to smallest for each enhancer class
    y_test_sorted = np.argsort(y_test, axis=0)[::-1]
    # get top predictions from y_test for specified enhancer class
    top_ix = y_test_sorted[:args.top_n,0 if args.enhancer=='Dev' else 1] 

    # get sequences to perform attribution analysis on
    examples = X_test[top_ix]

    # select background data for shap
    background_seqs = None
    if args.method == 'shap' and not args.dinuc_shuffle:
        # select a set of background examples to take an expectation over
        rng = np.random.default_rng(1234)
        background_seqs = X_test[rng.choice(X_test.shape[0], size=args.ref_size, replace=False, shuffle=False)]

    # collect cumsum of values for averaging 
    cumsum = 0

    # iterate through models in ensemble
    for i in range(args.n_mods):
        print(f'{args.method} analysis on model {i+1}/{args.n_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model 
        if args.evoaug:
            import evoaug_tf
            from evoaug_tf import evoaug, augment
            augment_list = [
                augment.RandomInsertionBatch(insert_min=0, insert_max=20),
                augment.RandomDeletion(delete_min=0, delete_max=30),
                augment.RandomTranslocationBatch(shift_min=0, shift_max=20)
            ]   
            model = utils.load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_DeepSTARR_finetune.h5"), 
                                                  input_shape=X_train[0].shape, 
                                                  augment_list=augment_list, 
                                                  config_file=args.config, 
                                                  epistemic=args.std)
        else:
            model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        
        # calculate gradients depending on which method is specified
        grads = utils.attribution_analysis(model, 
                                           examples, 
                                           args.method, 
                                           enhancer=args.enhancer, 
                                           head=args.head,
                                           ref_size=args.ref_size,
                                           background=background_seqs)

        # gradient correction
        grads -= np.mean(grads, axis=-1, keepdims=True)
        
        # track cumsum if calculating average across ensemble
        if args.average:
            cumsum += grads

        # save as npy file
        if args.method=='shap' and args.dinuc_shuffle:
            np.save(file=join(outdir, str(i+1) + "_top" + str(args.top_n) + f"_{args.enhancer}-{args.head}_shap_dinuc_shuffle.npy"),
                arr=grads)
        else:
            np.save(file=join(outdir, str(i+1) + "_top" + str(args.top_n) + f"_{args.enhancer}-{args.head}_{args.method}.npy"),
                    arr=grads)
    
    # calculate avg and save as npy file
    if args.average:
        avg_pred = cumsum/args.n_mods 
        if args.dinuc_shuffle:
            assert((args.dinuc_shuffle and args.method=='saliency') is not True) # make sure dinuc_shuffle is not set with saliency
            np.save(file=join(outdir, "avg_top" + str(args.top_n) + f"_{args.enhancer}-{args.head}_{args.method}_dinuc_shuffle.npy"), 
                arr=avg_pred)
        else:
            np.save(file=join(outdir, "avg_top" + str(args.top_n) + f"_{args.enhancer}-{args.head}_{args.method}.npy"), 
                    arr=avg_pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)