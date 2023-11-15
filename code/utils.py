import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd

def load_DeepSTARR_data(file):
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

    data.close()
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def downsample(X_train, y_train, p):
    '''
    randomly downsample training data 
    p = [0,1) determines proportion of training data to keep
    '''
    n_samples = X_train.shape[0]
    n_downsample = round(n_samples*p)
    ix = np.random.randint(0, n_samples+1, size=n_downsample)
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


