import matplotlib.pyplot as plt

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
    plt.xlabel(loss_fxn + ' epoch')
    plt.legend()
    # Add vertical line at minimum validation loss of combined dev and hk
    min_val_loss = min(history.history['val_loss'])
    plt.axvline(x=history.history['val_loss'].index(min_val_loss), color='red', linestyle='--')
    plt.savefig(out_fh)

