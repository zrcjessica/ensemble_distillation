import argparse
import utils
from os.path import basename
import numpy as np
from os.path import join

'''
for an ensemble or set of models, analyze attribution scores (saliency or shap)
if rmse flag set: calculates Euclidean distance as RMSE (must provide reference file)
if var flag set: calculates variance in attributions cores across models 
performs analysis for DeepSTARR test sequences selected by binning
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_dir", type=str,
                        help='path directory storing attribution analysis results for ensemble')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--quantile", type=float,
                        help='which quantile sequences to analyze')
    parser.add_argument("--nseqs", default=100,
                        help='number of seqs per quantile bin')
    parser.add_argument("--reference", type=str, default=None,
                        help='provide path to file containing reference values (required if rmse flag set)')
    parser.add_argument('--method', type=str,
                        help='what kind of attribution analysis was done (saliency/shap)')
    parser.add_argument("--rmse", action='store_true',
                        help='if set, calculate RMSE between attribution scores and reference')
    parser.add_argument("--var", action='store_true',
                        help='if set, calculate variance in attribution scores across models')
    args = parser.parse_args()
    return args

def main(args):
    # set output directory
    outdir = args.out
    if outdir is None:
        outdir = args.files_dir

    # get files
    attr_files, ref_file = utils.get_binned_attribution_files(args.files_dir, args.quantile, args.method, args.nseqs, args.reference)

    if args.rmse:
        # calculate rmse 
        ref_grad = np.load(ref_file) # load reference values
        for f in attr_files:
            model_ix = basename(f).split('_')[0]
            print(f'Calculating Euclidean distance for model {model_ix}')
            grad = np.load(f)
            distances = (np.mean(np.square((grad - ref_grad)), axis=(-1,-2)))**0.5
            # write to file 
            np.save(join(outdir, f"{model_ix}_q{args.quantile}_{args.nseqs}seqs_{args.method}_euclidean_dist.npy"), distances)
    
    if args.var:
        # calculate variance
        all_attr_scores = np.array([np.load(x) for x in attr_files])
        positional_var = np.var(all_attr_scores, axis=0) # get per nucleotide and sequence position var across models 
        seq_var = np.sum(positional_var, axis=(-1, -2)) # sum variances across nucleotides and sequences
        seq_std = np.sqrt(seq_var) # calculate stdev
        np.save(join(outdir, f"q{args.quantile}_{args.nseqs}seqs_{args.method}_stdev.npy"), seq_std)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)