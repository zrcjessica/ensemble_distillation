import argparse
import logomaker
import glob 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import basename
from os.path import join
import utils 


'''
- Outputs a multi-page PDF comparing attribution analysis results for all models in ensemble
- One logo (sequence) per page
- Includes average attribution map across ensemble at the bottom
- Must specify path to average attribution map and attribution analysis method (saliency/shap)
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_dir", type=str,
                        help='path directory storing attribution analysis results for ensemble')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--top_n", type=int, default=10,
                        help='how many logos to plot')
    parser.add_argument("--seq", nargs='+',
                        help='specify which sequence(s) to plot logos for as strings')
    parser.add_argument("--average", type=str, default=None,
                        help='provide path to average saliency analysis results')
    parser.add_argument('--method', type=str,
                        help='what kind of attribution analysis was done (saliency/shap)')
    args = parser.parse_args()
    return args

def main(args):

    # set output directory
    outdir = args.out
    if outdir is None:
        outdir = args.files_dir

    # get files
    # attr_files, avg_file = utils.get_saliency_files(args.saliency_dir, avg_file=args.average)
    attr_files, avg_file = utils.get_attribution_files(args.files_dir, args.method, args.average)

    # plot multipage pdf
    with PdfPages(join(outdir, f"top{args.top_n}_{args.method}_logos.pdf")) as pdf:
        for i in range(args.top_n):
            nrow = len(attr_files)+1
            fig, axs = plt.subplots(nrow,1)
            for f in attr_files:
                ix = int(basename(f).split("_")[0])
                print(f'model {ix}')
                attr_df = utils.parse_attribution_df(f, i)
                logomaker.Logo(attr_df, ax=axs[ix-1])
                axs[ix-1].set_title(f'Model {ix}')
                axs[ix-1].axis('off')
            # Plot average 
            attr_df = utils.parse_attribution_df(avg_file, i)
            logomaker.Logo(attr_df, ax=axs[len(attr_files)])
            axs[len(attr_files)].set_title('Ensemble average')
            axs[len(attr_files)].tick_params(left=False, right=False, labelleft=False,
                                                 labelbottom=False, bottom=False) 
            # format 
            fig.set_size_inches(10, 15)
            fig.suptitle(f'Sequence {i}')
            fig.tight_layout()
            pdf.savefig()

if __name__ == "__main__":
    args = parse_args()
    main(args)