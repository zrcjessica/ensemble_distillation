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

'''
run attribution analysis (saliency/shap) top n examples from test set (y_test) for an ensemble of DeepSTARR models
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help='path directory storing trained ensemble of models')
    parser.add_argument("--n_mods", type=int, 
                        help="number of models in ensemble")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--top_n", type=int, default=500,
                        help='how many of top predictions from test set to analyze')
    parser.add_argument('--enhancer', type=str, default='Dev',
                        help='which class of predictions to sort top predictions for')
    parser.add_argument("--average", action='store_true',
                        help='if set, calculate average saliency map across all models provided')
    parser.add_argument("--method", type=str,
                        help='saliency or shap; determines what method to use for attribution map')
    parser.add_argument("--dinuc_shuffle", action='store_true',
                        help='set if method=shap; if true, use dinucleotide shuffled seqs for shap reference per sequence')
    parser.add_argument("--ref_size", default=100, type=int,
                        help='set if method=shap; size of reference set')
    parser.add_argument("--std", action='store_true',
                        help='if true, model predicts standard deviation')
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

    # parse sequences to perform attribution analysis on
    examples = X_test[top_ix]
    # if args.method == 'saliency':
    #     # instantiate Variable class of examples to analyze
    #     examples = Variable(X_test[top_ix])

    # select background data for shap
    background_seqs = None
    if args.method == 'shap' and not args.dinuc_shuffle:
        # select a set of background examples to take an expectation over
        np.random.seed(seed=1234)
        print(type(X_test.shape[0]))
        print(type(args.ref_size))
        background_seqs = X_test[np.random.choice(X_test.shape[0], args.ref_size, replace=False)]

    # collect cumsum of values for averaging 
    cumsum = 0

    # iterate through models in ensemble
    for i in range(args.n_mods):
        print(f'ensemble {args.method} analysis on model {i+1}/{args.n_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model 
        model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        
        # calculate gradients depending on which method is specified
        grads = utils.attribution_analysis(model, examples, args.method, 
                                           enhancer=args.enhancer, ref_size=args.ref_size,
                                           background=background_seqs)

        # gradient correction
        grads -= np.mean(grads, axis=-1, keepdims=True)
        
        # track cumsum if calculating average across ensemble
        if args.average:
            cumsum += grads

        # save as npy file
        if args.dinuc_shuffle:
            np.save(file=join(outdir, str(i+1) + "_top" + str(args.top_n) + f"_shap_dinuc_shuffle.npy"),
                arr=grads)
        else:
            np.save(file=join(outdir, str(i+1) + "_top" + str(args.top_n) + f"_{args.method}.npy"),
                    arr=grads)
    
    # calculate avg and save as npy file
    if args.average:
        avg_pred = cumsum/args.n_mods 
        if args.dinuc_shuffle:
            np.save(file=join(outdir, "average_top" + str(args.top_n) + f"_{args.method}_dinuc_shuffle.npy"), 
                arr=avg_pred)
        else:
            np.save(file=join(outdir, "average_top" + str(args.top_n) + f"_{args.method}.npy"), 
                    arr=avg_pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)