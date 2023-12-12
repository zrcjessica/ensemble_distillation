import argparse
import logomaker
import glob 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import basename
from os.path import join
import utils 


'''
- Outputs a multi-page PDF comparing saliency analysis results for all models in ensemble
- One logo (sequence) per page
- Includes average saliency map across ensemble at the bottom
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliency_dir", type=str,
                        help='path directory storing saliency analysis results for ensemble')
    parser.add_argument("--out", type=str,
                        help='where to save results')
    parser.add_argument("--top_n", type=int, default=10,
                        help='how many logos to plot')
    parser.add_argument("--average", type=str, default=None,
                        help='provide path to average saliency analysis results')
    args = parser.parse_args()
    return args

def main(args):

    # set output directory
    outdir = args.out
    if outdir is None:
        outdir = args.saliency_dir

    # get files
    saliency_files, avg_file = utils.get_saliency_files(args.saliency_dir, avg_file=args.average)

    # plot multipage pdf
    with PdfPages(join(outdir, f"top{args.top_n}_saliency_logos.pdf")) as pdf:
        for i in range(args.top_n):
            nrow = len(saliency_files)+1
            fig, axs = plt.subplots(nrow,1)
            for f in saliency_files:
                ix = int(basename(f).split("_")[0])
                print(f'model {ix}')
                saliency_df = utils.parse_saliency_df(f, i)
                logomaker.Logo(saliency_df, ax=axs[ix-1])
                axs[ix-1].set_title(f'Model {ix}')
                axs[ix-1].axis('off')
            # Plot average 
            saliency_df = utils.parse_saliency_df(avg_file, i)
            logomaker.Logo(saliency_df, ax=axs[len(saliency_files)])
            axs[len(saliency_files)].set_title('Ensemble average')
            axs[len(saliency_files)].tick_params(left=False, right=False, labelleft=False,
                                                 labelbottom=False, bottom=False) 
            # axs[len(saliency_files)].axis('off')
            # format 
            fig.set_size_inches(10, 15)
            fig.suptitle(f'Sequence {i}')
            fig.tight_layout()
            pdf.savefig()

if __name__ == "__main__":
    args = parse_args()
    main(args)