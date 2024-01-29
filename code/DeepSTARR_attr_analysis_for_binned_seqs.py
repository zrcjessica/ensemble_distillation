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
bin sequences from DeepSTARR test set by quantile
run attribution analysis (saliency/shap) for sequences sampled from quantile bins for replicates of DeepSTARR models
can average attribution maps across a set of replicates (e.g. ensemble)
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
    parser.add_argument("--seqs", type=str,
                        help='table of sequence ixs per bin; if provided, skips binning step')
    parser.add_argument("--nbins", default=4,
                        help='number of bins')
    parser.add_argument("--nseqs", type=int, default=100,
                        help='how many seqs to select from each bin')
    parser.add_argument('--enhancer', type=str, default='Dev',
                        help='which class of predictions to sort top predictions for')
    parser.add_argument("--average", action='store_true',
                        help='if set, calculate average attribution map across all models provided')
    parser.add_argument("--method", type=str,
                        help='saliency or shap; determines what method to use for attribution map')
    parser.add_argument("--ref_size", default=1000, type=int,
                        help='size of reference set for shap')
    parser.add_argument("--std", action='store_true',
                        help='if true, model predicts standard deviation')
    parser.add_argument("--predict", action='store_true',
                        help='if set, predict with models for selected seqs')
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data, std=args.std)

    # set output directory
    outdir = args.out
    if outdir is None:
        outdir=args.model_dir

    bins_seqs_dict = {}
    if args.seqs is not None:
        # load sequences to analyze from file provided
        seqs_df = pd.read_csv(args.seqs)
        bins_seqs_dict = seqs_df.to_dict('list')
    else:
        # quantile bin seqs based on y_test for selected enhancer class
        quantiles = np.arange(1, 0, -(1/args.nbins))[::-1]
        bins = pd.qcut(y_test[:,0 if args.enhancer=='Dev' else 1], 4, labels=quantiles, retbins=True)

        # record intervals corresponding to each quantile
        intervals = [f'({bins[1][i]:.3f},{bins[1][i+1]}:.3f]' for i in range(args.nbins)]
        intervals_df = pd.DataFrame({'quantiles': quantiles, 'intervals': intervals})
        intervals_df.to_csv(join(outdir, f"{args.enhancer}_test_quantiles_and_intervals.csv"))

        # sample seqs from bins 
        rng = np.random.default_rng(1)
        bins_seqs_dict = {q:rng.choice(np.where(bins[0]==q)[0], size=args.nseqs, replace=False) for q in quantiles}
        # record 
        bins_seqs_df = pd.DataFrame.from_dict(bins_seqs_dict)
        bins_seqs_df.to_csv(join(outdir, f"{args.nseqs}_binned_seqs.csv"))

    # select background data for shap
    background_seqs = None
    if args.method == 'shap':
        # select a set of background examples to take an expectation over
        np.random.seed(seed=1234)
        background_seqs = X_test[np.random.choice(X_test.shape[0], args.ref_size, replace=False)]

    for q, seqs_ix in bins_seqs_dict.items():
        print(f'{q} quantile bin')

        # parse sequences to perform attribution analysis on
        examples = X_test[seqs_ix]

        # collect cumsum of values for averaging 
        cumsum = 0

        # iterate through models in ensemble
        for i in range(args.n_mods):
            print(f'{args.method} analysis on model {i+1}/{args.n_mods}')

            # clear history
            keras.backend.clear_session()
            gc.collect()

            # load model 
            model = load_model(join(args.model_dir, f"{i+1}_DeepSTARR.h5"))
            
            # calculate gradients depending on which method is specified - if no background seqs provided, uses dinuc shuffle
            grads = utils.attribution_analysis(model, examples, args.method, 
                                            enhancer=args.enhancer, 
                                            background=background_seqs)

            # gradient correction
            grads -= np.mean(grads, axis=-1, keepdims=True)
            
            # track cumsum if calculating average across ensemble
            if args.average:
                cumsum += grads

            # save as npy file
            np.save(file=join(outdir, f"{i+1}_q{q}_{args.nseqs}seqs_{args.method}.npy"), arr=grads)

            # predict and save
            if args.predict:
                print('making predictions with model')
                np.save(file=join(outdir, f"{i+1}_q{q}_{args.nseqs}seqs_preds.npy"), arr=model.predict(examples))

        # calculate avg and save as npy file
        if args.average:
            avg_pred = cumsum/args.n_mods 
            np.save(file=join(outdir, f"average_q{q}_{args.nseqs}seqs_{args.method}.npy"), arr=avg_pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)