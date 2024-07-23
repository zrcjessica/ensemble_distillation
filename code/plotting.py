import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
import logomaker
from matplotlib.backends.backend_pdf import PdfPages

def plot_loss(history, out_fh, loss_fxn=None):
    '''
    plot train and val loss curves per epoch
    history is original history object returned by keras
    '''
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    if loss_fxn is not None:
        plt.title(loss_fxn + ' loss')
        plt.ylabel(loss_fxn + ' loss')
    else:
        plt.title("Loss")
        plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.legend()
    # Add vertical line at minimum validation loss of combined dev and hk
    min_val_loss = min(history.history['val_loss'])
    plt.axvline(x=history.history['val_loss'].index(min_val_loss), color='red', linestyle='--')
    plt.savefig(out_fh)

def prediction_scatterplot(pred, true, colnames, outfh):
    '''
    make a scatterplot of predictions versus true values
    '''
    pred_df = pd.DataFrame(pred, columns=colnames)
    pred_df['ix'] = np.arange(pred_df.shape[0])
    pred_melt = pd.melt(pred_df, id_vars='ix', var_name='output', value_name='pred')
    true_df = pd.DataFrame(true, columns=colnames)
    true_df['ix'] = np.arange(true_df.shape[0])
    true_melt = pd.melt(true_df, id_vars='ix', var_name='output', value_name='true')
    plotdf = pd.merge(pred_melt, true_melt, how='left')
    fig = sns.relplot(
        data=plotdf,
        x="true", y="pred",
        col="output",
        kind="scatter",
        col_wrap=2,
        facet_kws={'sharey': False, 'sharex': False}
    )
    fig.savefig(outfh, dpi=600)

def plot_logos(scores, fh, n_seqs=10):
    '''
    for a given model, plot attribution scores for top n seqs
    '''
    with PdfPages(fh) as pdf:
        fig, axs = plt.subplots(n_seqs,1)
        for seq_ix in range(n_seqs):
             # iterate through seqs 
            grad = scores[seq_ix,:,:]
            saliency_df = pd.DataFrame(grad)
            saliency_df.rename(columns={0:'A', 1:'C', 2:'G', 3:'T'}, inplace=True)
            logomaker.Logo(saliency_df, ax=axs[seq_ix])
        fig.set_size_inches(7, 10)
        fig.tight_layout()
        pdf.savefig()
