import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd
from os.path import basename
from os.path import join
import glob
import shap # shap must be imported before keras
from tensorflow import GradientTape
# from tensorflow import Variable
# import keras

def load_DeepSTARR_data(file, get_idx=False):
    '''
    load Train/Test/Val data from DeepSTARR h5
    '''
    data = h5py.File(file, 'r')
    
    # test
    X_train = np.array(data['X_Train'])
    y_train = np.array(data['y_Train'])

    # train
    X_test = np.array(data['X_Test'])
    y_test = np.array(data['y_Test'])

    # validation
    X_val = np.array(data['X_Val'])
    y_val = np.array(data['y_Val'])

    # get idx
    if get_idx:
        idx_train = np.array(data['idx_Train'])
        idx_test = np.array(data['idx_Test'])
        idx_val = np.array(data['idx_Val'])
        data.close()
        return X_train, y_train, X_test, y_test, X_val, y_val, idx_train, idx_test, idx_val
    else:
        data.close()
        return X_train, y_train, X_test, y_test, X_val, y_val

def downsample(X_train, y_train, p):
    '''
    randomly downsample training data 
    p = [0,1) determines proportion of training data to keep
    '''
    n_samples = X_train.shape[0]
    n_downsample = round(n_samples*p)
    ix = np.random.randint(0, n_samples, size=n_downsample)
    return X_train[ix,:], y_train[ix,:]

def evaluate_performance(y_pred, y_truth):
    '''
    calculate MSE, Spearman + Pearson corr between predicted/ground truth values
    '''
    mse = mean_squared_error(y_truth, y_pred)
    pearson = pearsonr(y_truth, y_pred)[0]
    spearman = spearmanr(y_truth, y_pred)[0]
    return [mse, pearson, spearman]

def summarise_DeepSTARR_performance(y_pred, y_truth):
    '''
    calculate MSE, Spearman + Pearson corr on test data
    return summary for Dev + Hk as data frame
    '''

    performance_dict = {}
    # calculate metrics for Dev
    performance_dict['Dev'] = evaluate_performance(y_pred[:,0], y_truth[:,0])
    # calculate metrics for Hk
    performance_dict['Hk'] = evaluate_performance(y_pred[:,1], y_truth[:,1])
    summary = pd.DataFrame(performance_dict)
    summary['metric'] = ['MSE', 'Pearson', 'Spearman']
    return summary

def get_saliency_files(dir, avg_file=None):
    '''
    returns a list of all files containing saliency analysis results for an ensemble
    if average is None (no average saliency file provided), assumes it is in the same directory
    '''
    if avg_file is None:
        avg_file = glob.glob(join(dir, "average*saliency.npy"))[0]
    saliency_files = glob.glob(join(dir, "*saliency.npy"))
    # check if avg file is here
    if avg_file in saliency_files:
        saliency_files = list(set(saliency_files) - set([avg_file]))
    return saliency_files, avg_file 

def parse_saliency_df(grad_file, i):
    '''
    given a gradient tensor, returns a dataframe for plotting with logomaker
    takes .npy file as input
    '''
    grad = np.load(grad_file)
    saliency_df = pd.DataFrame(grad[i])
    saliency_df.rename(columns={0:'A', 1:'C', 2:'G', 3:'T'}, inplace=True)
    return saliency_df

def attribution_analysis(model, seqs, method, enhancer='Dev', background=None):
    '''
    returns attribution maps for model and seqs based on method (saliency/shap)
    if method=shap, must provide background seqs
    by default, returns attributions cores for Dev enhancers, can also specify Hk
    '''
    print(f'calculating attribution maps with {method}')
    if method == 'saliency':
        # saliency analysis
        with GradientTape() as tape:
            preds = model(seqs, training=False)
            loss = preds[:,0 if enhancer=='Dev' else 1]
        return tape.gradient(loss, seqs)
    else:
        # DeepExplainer 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
        explainer_dev = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), data=background)
        return explainer_dev.shap_values(seqs)[0]
    
# def summarise_ensemble_performance(files, downsample=1):
#     '''
#     summarises results of all individual models in an ensemble
#     takes a list of the output files for performance of models in ensemble
#     '''
    
#     results_list = []

#     for f in files:
#         df = pd.read_csv(f)
#         model_ix = int(basename(f).split('_')[0])
#         df['model_ix'] = model_ix
#         if 'metric' not in df.columns:
#             df['metric'] = ['MSE', 'Pearson', 'Spearman']
#         results_list.append(df)

#     # combine as one df
#     all_results = pd.concat(results_list)
#     all_results['model_ix'] = all_results['model_ix'].astype('int32')
#     all_results.sort_values(by = 'model_ix', inplace=True)
#     all_results.reset_index(drop=True, inplace=True)
#     all_results['downsample'] = downsample
#     return all_results