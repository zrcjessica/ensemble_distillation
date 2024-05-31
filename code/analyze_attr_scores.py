import argparse
import utils
from os.path import basename
import numpy as np
from os.path import join

'''
for an ensemble or set of models, analyze attribution scores (saliency or shap)
if rmse flag set: calculates Euclidean distance as RMSE (must provide reference file)
if var flag set: calculates variance in attribution scores across models 
code is currently designed for analyzing attribution scores for top 500 (Dev enhancer) DeepSTARR test seqs
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_dir", type=str,
                        help='path directory storing attribution analysis results for ensemble')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--reference", type=str, default=None,
                        help='provide path to file containing reference values (required if rmse flag set)')
    parser.add_argument('--method', type=str,
                        help='what kind of attribution analysis was done (saliency/shap)')
    parser.add_argument("--rmse", action='store_true',
                        help='if set, calculate RMSE between attribution scores and reference')
    parser.add_argument("--var", action='store_true',
                        help='if set, calculate variance in attribution scores across models')
    parser.add_argument("--top_n", type=int, default=500,
                        help='how many of top predictions from test set to analyze')
    parser.add_argument('--enhancer', type=str, default='Dev',
                        help='which class of predictions to sort top predictions for')
    args = parser.parse_args()
    return args

def main(args):
    # set output directory
    outdir = args.out
    if outdir is None:
        outdir = args.files_dir

    # get files
    attr_files, ref_file = utils.get_attribution_files(args.files_dir, args.method, args.enhancer, args.top_n, args.reference)

    if args.rmse:
        # calculate rmse 
        ref_grad = np.load(ref_file) # load reference values
        for f in attr_files:
            model_ix = basename(f).split('_')[0]
            print(f'Calculating Euclidean distance for model {model_ix}')
            grad = np.load(f)
            distances = (np.mean(np.square((grad - ref_grad)), axis=(-1,-2)))**0.5
            # write to file 
            np.save(join(outdir, f"{model_ix}_top{args.top_n}_{args.enhancer}_{args.method}_euclidean_dist.npy"), distances)
    
    if args.var:
        # calculate variance
        all_attr_scores = np.array([np.load(x) for x in attr_files])
        positional_var = np.var(all_attr_scores, axis=0) # get per nucleotide and sequence position var across models 
        seq_var = np.sum(positional_var, axis=(-1, -2)) # sum variances across nucleotides and sequences
        seq_std = np.sqrt(seq_var) # calculate stdev
        np.save(join(outdir, f"top{args.top_n}_{args.enhancer}_{args.method}_stdev.npy"), seq_std)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)