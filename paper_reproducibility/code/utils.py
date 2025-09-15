import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd
from os.path import basename
from os.path import join
import glob
import shap # shap must be imported before keras
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow import Variable
import keras
import gc
# np.random.seed(1)
from keras.optimizers import Adam
from model_zoo import DeepSTARR, lentiMPRA
import evoaug_tf
from evoaug_tf import evoaug, augment
import yaml

def load_lentiMPRA_data(file, epistemic=False):
    '''
    Load lentiMPRA train, test, and validation datasets from an HDF5 file.

    Parameters:
    ----------
    file : str
        Path to the HDF5 file containing lentiMPRA data.
    epistemic : bool, optional
        If True, removes the last column of the target data, assuming it contains standard deviations 
        for aleatoric uncertainty in epistemic models (which is not used). Default is False.

    Returns:
    -------
    tuple
        X_train, y_train, X_test, y_test, X_val, y_val: NumPy arrays for input and target data across 
        training, test, and validation datasets.
    '''
    data = h5py.File(file, 'r')

    # train
    X_train = np.array(data['Train']['X'])
    y_train = np.array(data['Train']['y'])

    # test
    X_test = np.array(data['Test']['X'])
    y_test = np.array(data['Test']['y'])

    # val
    X_val = np.array(data['Val']['X'])
    y_val = np.array(data['Val']['y'])

    if epistemic:
        # file w/ data for epistemic models has std for aleatoric prediction in last col; do not return
        y_train = y_train[:,:-1]
        y_test = y_test[:,:-1]
        y_val = y_val[:,:-1]

    data.close()
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def load_DeepSTARR_data(file, std=False):
    '''
    Load train, test, and validation datasets from a DeepSTARR HDF5 file, 
    with options to include standard deviation (epistemic uncertainty) values.

    Parameters:
    ----------
    file : str
        Path to the HDF5 file containing DeepSTARR data.
    std : bool, optional
        If True, appends standard deviation values to target (y) data for train, test, and validation sets. 
        Default is False.

    Returns:
    -------
    tuple or dict
        X_train, y_train, X_test, y_test, X_val, y_val
    '''
    if 'hierarchical' in file:
        print('loading data from h5 file with hierarchical structure')
        X_train, y_train, X_test, y_test, X_val, y_val = load_DeepSTARR_data_hierarchical(file, std)
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
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

        # add standard deviation data 
        if std:
            y_train = np.array(data['ensemble_mean']) # ensemble avg. of train seqs should be in h5
            y_train = np.append(y_train, np.array(data['std_Train']), axis=1)
            y_test = np.append(y_test, np.array(data['std_Test']), axis=1)
            y_val = np.append(y_val, np.array(data['std_Val']), axis=1)

        data.close()
        return X_train, y_train, X_test, y_test, X_val, y_val
    
def load_DeepSTARR_data_hierarchical(file, std=False):
    '''
    Load train, test, and validation datasets from a hierarchically structured DeepSTARR HDF5 file.

    Parameters:
    ----------
    file : str
        Path to the HDF5 file containing DeepSTARR data in a hierarchical structure.
    std : bool, optional
        If True, appends standard deviation values to target (y) data for train, test, and validation sets. 
        Default is False.

    Returns:
    -------
    tuple
        X_train, y_train, X_test, y_test, X_val, y_val: NumPy arrays for input and target data across 
        training, test, and validation datasets.
    '''
    data = h5py.File(file, 'r') 
    
    # train
    X_train = np.array(data['Train']['X'])
    y_train = np.array(data['Train']['y'])

    # test
    X_test = np.array(data['Test']['X'])
    y_test = np.array(data['Test']['y'])

    # validation
    X_val = np.array(data['Val']['X'])
    y_val = np.array(data['Val']['y'])

    # add standard deviation data 
    if std:
        y_train = np.array(data['Train']['ensemble_mean']) # ensemble avg. of train seqs should already be in h5
        y_train = np.append(y_train, np.array(data['Train']['std']), axis=1)
        y_test = np.append(y_test, np.array(data['Test']['std']), axis=1)
        y_val = np.append(y_val, np.array(data['Val']['std']), axis=1)

    data.close()
    return X_train, y_train, X_test, y_test, X_val, y_val 

def downsample(X_train, y_train, rng, p, return_ix=False):
    '''
    Randomly downsample the training data by a specified proportion.

    Parameters:
    ----------
    X_train : np.array
        Input training data.
    y_train : np.array
        Target training data.
    rng : np.random.Generator
        Random number generator for reproducible downsampling.
    p : float
        Proportion of training data to keep, where 0 <= p < 1.
    return_ix : bool, optional
        If True, returns only the indices of selected samples instead of downsampled data.
        Default is False.

    Returns:
    -------
    tuple or np.array
        If return_ix is True, returns a 1D array of indices for downsampled data.
        Otherwise, returns the downsampled X_train and y_train arrays.
    '''
    n_samples = X_train.shape[0]
    n_downsample = round(n_samples*p)
    ix = rng.choice(n_samples, size=n_downsample, replace=False, shuffle=False)
    if return_ix:
        return ix
    else:
        if len(y_train.shape)==1:
            return X_train[ix,:], y_train[ix]
        else:
            return X_train[ix,:], y_train[ix,:]

def evaluate_performance(y_pred, y_truth):
    '''
    Calculate the Mean Squared Error (MSE), Pearson correlation, and Spearman correlation 
    between predicted and ground truth values.

    Parameters:
    ----------
    y_pred : np.array
        Predicted values.
    y_truth : np.array
        Ground truth values.

    Returns:
    -------
    list
        A list containing the MSE, Pearson correlation coefficient, and Spearman correlation coefficient.
    '''
    mse = mean_squared_error(y_truth, y_pred)
    pearson = pearsonr(y_truth, y_pred)[0]
    spearman = spearmanr(y_truth, y_pred)[0]
    return [mse, pearson, spearman]

def summarise_lentiMPRA_performance(y_pred, y_truth, celltype, aleatoric=False, epistemic=False):
    '''
    Calculate MSE, Pearson, and Spearman correlations on test data for a specified cell type for lentiMPRA data, 
    and return a summary as a DataFrame. Can include aleatoric and/or epistemic uncertainty metrics if specified.

    Parameters:
    ----------
    y_pred : np.array
        Predicted values with columns for base predictions and optional uncertainty predictions.
    y_truth : np.array
        Ground truth values with columns corresponding to y_pred.
    celltype : str
        Cell type name (K562 or HepG2), used as a key in the output DataFrame.
    aleatoric : bool, optional
        If True, include aleatoric uncertainty performance metrics.
    epistemic : bool, optional
        If True, include epistemic uncertainty performance metrics.

    Returns:
    -------
    pd.DataFrame
        DataFrame summarizing MSE, Pearson, and Spearman correlation metrics for base predictions
        and any specified uncertainties.
    '''
    performance_dict = {}
    # calculate metrics for celltype
    performance_dict[celltype] = evaluate_performance(y_pred[:,0], y_truth[:,0])
    # calculate metrics for standard deviation predictions
    if aleatoric and epistemic:
        performance_dict[f'{celltype}-aleatoric'] = evaluate_performance(y_pred[:,1], y_truth[:,1])
        performance_dict[f'{celltype}-epistemic'] = evaluate_performance(y_pred[:,2], y_truth[:,2])
    elif aleatoric:
        performance_dict[f'{celltype}-aleatoric'] = evaluate_performance(y_pred[:,1], y_truth[:,1])
    elif epistemic:
        performance_dict[f'{celltype}-epistemic'] = evaluate_performance(y_pred[:,1], y_truth[:,1])
    summary = pd.DataFrame(performance_dict)
    summary['metric'] = ['MSE', 'Pearson', 'Spearman']
    return summary

def summarise_DeepSTARR_performance(y_pred, y_truth, std=False):
    '''
    Calculate MSE, Pearson, and Spearman correlations on test data for "Dev" and "Hk" targets, 
    and return a summary as a DataFrame. Can include uncertainty metrics if specified.

    Parameters:
    ----------
    y_pred : np.array
        Predicted values with columns for "Dev" and "Hk" predictions, and optional uncertainty predictions.
    y_truth : np.array
        Ground truth values with columns corresponding to y_pred.
    std : bool, optional
        If True, include uncertainty performance metrics for "Dev" and "Hk".

    Returns:
    -------
    pd.DataFrame
        DataFrame summarizing MSE, Pearson, and Spearman correlation metrics for "Dev", "Hk", 
        and any specified uncertainties.
    '''
    performance_dict = {}
    # calculate metrics for Dev
    performance_dict['Dev'] = evaluate_performance(y_pred[:,0], y_truth[:,0])
    # calculate metrics for Hk
    performance_dict['Hk'] = evaluate_performance(y_pred[:,1], y_truth[:,1])
    # calculate metrics for standard deviation predictions
    if std:
        performance_dict['Dev-std'] = evaluate_performance(y_pred[:,2], y_truth[:,2])
        performance_dict['Hk-std'] = evaluate_performance(y_pred[:,3], y_truth[:,3])
    summary = pd.DataFrame(performance_dict)
    summary['metric'] = ['MSE', 'Pearson', 'Spearman']
    return summary

def get_attribution_files(dir, method, enhancer='Dev', top_n=500, avg_file=None):
    """
    Retrieves a list of files containing attribution analysis results for a model ensemble.

    This function searches for attribution files in the specified directory (`dir`) for 
    a given method (e.g., SHAP or saliency) and enhancer type. It returns a list of 
    attribution files for the top_n sequences (default=500) as well as the path 
    to an average attribution file. If an average file is not provided, the function 
    assumes it follows a specific naming pattern and searches for it in `dir`.

    Parameters:
    - dir (str): The directory path containing attribution files.
    - method (str): The attribution method used (e.g., 'shap' or 'saliency').
    - enhancer (str): The enhancer type (default is 'Dev').
    - top_n (int): The number of top sequences to consider (default is 500).
    - avg_file (str, optional): The file path for the average attribution file. 
                                If None, the function will search for a file with 
                                the naming pattern `avg_top{top_n}_{enhancer}-mean_{method}.npy`.

    Returns:
    - tuple:
        - attr_files (list of str): List of file paths for individual attribution analysis results.
        - avg_file (str): File path of the average attribution file (relative to which RMSE will be calculated).
    """
    if avg_file is None:
        avg_file = join(dir, f"avg_top{top_n}_{enhancer}-mean_{method}.npy")
    attr_files = glob.glob(join(dir, f"*_top{top_n}_{enhancer}-mean_{method}.npy"))
    
    # Exclude the average file if present in the attribute files list
    if avg_file in attr_files:
        attr_files = list(set(attr_files) - set([avg_file]))
        
    return attr_files, avg_file


def get_lentiMPRA_attribution_files(dir, method, celltype, top_n=1000, avg_file=None):
    '''
    returns a list of all files containing attribution analysis results for an ensemble
    if average is None (no average attribution map provided), assumes it is in the same directory
    must supply method (shap/saliency) 
    gets attr. scores for top 500 (Dev enhancers) DeepSTARR test seqs 
    '''
    if avg_file is None:
        # avg_file = join(dir, f"average*{method}.npy")
        avg_file = join(dir, f"avg_top{top_n}_{celltype}-mean_{method}.npy")
    attr_files = glob.glob(join(dir, f"*_top{top_n}_{celltype}-mean_{method}.npy"))
    # check if avg file is here
    if avg_file in attr_files:
        attr_files = list(set(attr_files) - set([avg_file]))
    return attr_files, avg_file 

# def parse_attribution_df(grad_file, i):
#     '''
#     given a gradient tensor, returns a dataframe for plotting with logomaker
#     takes .npy file as input
#     '''
#     grad = np.load(grad_file)
#     attr_df = pd.DataFrame(grad[i])
#     attr_df.rename(columns={0:'A', 1:'C', 2:'G', 3:'T'}, inplace=True)
#     return attr_df

# taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    Parameters
    ----------
    seq : str or ndarray
        either a string of length L, or an L x D NumPy array of one-hot encodings
    num_shufs : int
        the number of shuffles to create, N; if unspecified, only one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles

    Returns
    -------
    list (if 'seq' is string)
        List of N strings of length L, each one being a shuffled version of 'seq'
        
    ndarray (if 'seq' is ndarray)
        ndarray of shuffled versions of 'seq' (shape=(N,L,D)), also one-hot encoded
        If 'num_shufs' is not specified, then the first dimension of N will not be present
        (i.e. a single string will be returned, or an LxD array).
    """
    def string_to_char_array(seq):
        """
        Converts an ASCII string to a NumPy array of byte-long ASCII codes.
        e.g. "ACGT" becomes [65, 67, 71, 84].
        """
        return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


    def char_array_to_string(arr):
        """
        Converts a NumPy array of byte-long ASCII codes into an ASCII string.
        e.g. [65, 67, 71, 84] becomes "ACGT".
        """
        return arr.tostring().decode("ascii")


    def one_hot_to_tokens(one_hot):
        """
        Converts an L x D one-hot encoding into an L-vector of integers in the range
        [0, D], where the token D is used when the one-hot encoding is all 0. This
        assumes that the one-hot encoding is well-formed, with at most one 1 in each
        column (and 0s elsewhere).
        """
        tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
        seq_inds, dim_inds = np.where(one_hot)
        tokens[seq_inds] = dim_inds
        return tokens


    def tokens_to_one_hot(tokens, one_hot_dim):
        """
        Converts an L-vector of integers in the range [0, D] to an L x D one-hot
        encoding. The value `D` must be provided as `one_hot_dim`. A token of D
        means the one-hot encoding is all 0s.
        """
        identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
        return identity[tokens]

    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")


    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]

def _pad_end(x, insert_max=20):
        """Add random DNA padding of length insert_max to the end of each sequence in batch.
        for saliency analysis with EvoAug models"""

        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)
        p = tf.ones((A,)) / A
        padding = tf.transpose(tf.gather(tf.eye(A), tf.random.categorical(tf.math.log([p] * insert_max), N)), perm=[1,0,2])

        half = int(insert_max/2)
        x_padded = tf.concat([padding[:,:half,:], x, padding[:,half:,:]], axis=1)
        return x_padded
    
def attribution_analysis(model, seqs, method, enhancer='Dev', head='mean', ref_size=100, background=None, evoaug=False):
    '''
    Performs attribution analysis on DeepSTARR output heads 

    Parameters
    ----------
    model : keras.engine.functional.Functional
        model to perform attribution analysis on 
    seqs : ndarray
        one hot encoded sequences (shape=(N,249,4)) that attribution maps will be calculated w.r.t 
    method : str 
        'shap' or 'saliency'; determines whether DeepSHAP or saliency maps will be used to calculate attribution maps 
    enhancer : str
        'Dev' or 'Hk'; which DeepSTARR promoter class to evaluate
    head : str
        'mean', 'std', or 'logvar'; determines which model ouptut head to calculate attribution map for (logvar will be treated the same as std)
    ref_size : int
        Number of shuffles to perform for dinuc_shuffle to generate DeepSHAP background seqs; default=100
    background : ndarray
        (optional) one hot encoded background sequences for DeepSHAP, will be generated with dinuc_shuffle if not provided; default=None
    evoaug : boolean
        set true if model was trained with evoaug, pads input seqs accordingly; default=False
    

    Returns
    -------
    ndarray 
        ndarray containing attribution maps for specified output head calculated with specified method (shape=(N,249,4))
    '''
    print(f'calculating attribution maps with {method}')
    if evoaug and model.insert_max != 0:
        seqs = _pad_end(seqs)
    if method == 'saliency':
        # saliency analysis
        seqs = Variable(seqs, dtype='float32')
        with GradientTape() as tape:
            preds = model(seqs, training=False)
            if enhancer=='Dev':
                loss = preds[:, 0 if head=='mean' else 2]
            else: # Hk 
                loss = preds[:, 1 if head=='mean' else 3]
            # loss = preds[:,0 if enhancer=='Dev' else 1]
        mean_saliency = tape.gradient(loss, seqs)
        return mean_saliency
    else:
        # DeepExplainer 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
        if background is None:
            # use dinuc shuffled bg seqs
            print('using dinucleotide shuffled sequences as shap reference')
            rng = np.random.default_rng(1)
            shap_values_arr = np.zeros(seqs.shape)
            for i in range(seqs.shape[0]):
                print(f'running DeepExplainer on sequence {i+1}/{seqs.shape[0]}')
                # clear history
                keras.backend.clear_session()
                gc.collect()
                seq = seqs[i]
                bg = dinuc_shuffle(seq, num_shufs=ref_size, rng=rng) # can we remove this feature if random selection works fine?
                explainer_dev = shap.DeepExplainer((model.input, model.output), data=bg)
                if enhancer=='Dev':
                    shap_values_arr[i] = np.squeeze(explainer_dev.shap_values(np.expand_dims(seq, axis=0))[0 if head=='mean' else 2], axis=0)
                else: # Hk 
                    shap_values_arr[i] = np.squeeze(explainer_dev.shap_values(np.expand_dims(seq, axis=0))[1 if head=='mean' else 3], axis=0)
                # shap_values_arr[i] = np.squeeze(explainer_dev.shap_values(np.expand_dims(seq, axis=0))[0 if enhancer=='Dev' else 1], axis=0)
            return shap_values_arr
        else:
            # use provided background seqs
            print('performing shap analysis without provided background seqs')
            explainer_dev = shap.DeepExplainer((model.input, model.output), data=background)
            if enhancer=='Dev':
                return explainer_dev.shap_values(seqs)[0 if head=='mean' else 2]
            else: # Hk
                return explainer_dev.shap_values(seqs)[1 if head=='mean' else 3]
            # return explainer_dev.shap_values(seqs)[0 if enhancer=='Dev' else 1]

def lentiMPRA_attribution_analysis(model, seqs, method, head='mean', ref_size=100, background=None, evoaug=False): 
    '''
    Performs attribution analysis on ResidualBind output heads 

    Parameters
    ----------
    model : keras.engine.functional.Functional
        model to perform attribution analysis on 
    seqs : ndarray
        one hot encoded sequences (shape=(N,249,4)) that attribution maps will be calculated w.r.t 
    method : str 
        'shap' or 'saliency'; determines whether DeepSHAP or saliency maps will be used to calculate attribution maps 
        note: shap doesn't work
    head : str
        'mean', 'aleatoric', or 'epistemic'; determines which model ouptut head to calculate attribution map for
    ref_size : int
        Number of shuffles to perform for dinuc_shuffle to generate DeepSHAP background seqs; default=100
    background : ndarray
        (optional) one hot encoded background sequences for DeepSHAP, will be generated with dinuc_shuffle if not provided; default=None
    evoaug : boolean
        set true if model was trained with evoaug, pads input seqs accordingly; default=False
    

    Returns
    -------
    ndarray 
        ndarray containing attribution maps for specified output head calculated with specified method (shape=(N,249,4))
    '''
    print(f'calculating attribution maps with {method}')
    if evoaug and model.insert_max != 0:
        seqs = _pad_end(seqs)
    if method == 'saliency':
        # saliency analysis
        seqs = Variable(seqs, dtype='float32')
        with GradientTape() as tape:
            preds = model(seqs, training=False)
            if head=='mean':
                loss = preds[:,0]
            elif head=='aleatoric':
                loss = preds[:,1]
            elif head=='epistemic':
                loss = preds[:,2]
        mean_saliency = tape.gradient(loss, seqs)
        return mean_saliency
    else:
        # DeepExplainer 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
        if background is None:
            # use dinuc shuffled bg seqs
            print('using dinucleotide shuffled sequences as shap reference')
            rng = np.random.default_rng(1)
            shap_values_arr = np.zeros(seqs.shape)
            for i in range(seqs.shape[0]):
                print(f'running DeepExplainer on sequence {i+1}/{seqs.shape[0]}')
                # clear history
                keras.backend.clear_session()
                gc.collect()
                seq = seqs[i]
                bg = dinuc_shuffle(seq, num_shufs=ref_size, rng=rng) # can we remove this feature if random selection works fine?
                explainer_dev = shap.DeepExplainer((model.input, model.output), data=bg)
                shap_values_arr[i] = np.squeeze(explainer_dev.shap_values(np.expand_dims(seq, axis=0))[0 if head=='mean' else 1 if head=='aleatoric' else 2], axis=0)
            return shap_values_arr
        else:
            # use provided background seqs
            print('performing shap analysis without provided background seqs')
            explainer_dev = shap.DeepExplainer((model.input, model.output), data=background)
            return explainer_dev.shap_values(seqs)[0 if head=='mean' else 1 if head=='aleatoric' else 2]
        

def load_model_from_weights(weights, input_shape, config_file, epistemic, with_evoaug=True, augment_list=None):
    '''
    load DeepSTARR model from weights 
    
    Parameters
    ----------
    weights : str
        path to .h5 file containing model weigts
    input_shape : tuple
        provide shape of model input
    config_file : str
        path to model config file (yaml)        
    epistemic : boolean
        set true if model has epistemic uncertainty output head
    with_evoaug : boolean
        set true if model was trained with EvoAug
    augment_list : list
        optional, if model was trained with EvoAug, provide list of augmentations used for training; required if with_evoaug=True 
    
    Returns
    -------
    keras.engine.functional.Functional 
        DeepSTARR model loaded from weights
    '''
    config = yaml.safe_load(open(config_file, 'r'))
    model = None
    if with_evoaug:
        model = evoaug.RobustModel(DeepSTARR, input_shape=input_shape, augment_list=augment_list, max_augs_per_seq=1, hard_aug=True, config=config, epistemic=epistemic)
    else:
        model = DeepSTARR(input_shape, config, epistemic)
    model.compile(optimizer=Adam(learning_rate=config['optim_lr']),
                  loss=config['loss_fxn'])
    model.load_weights(weights)
    return model

def load_lentiMPRA_from_weights(weights, input_shape, config_file, aleatoric=False, epistemic=False, with_evoaug=True, augment_list=None):
    '''
    load ResidualBind model from weights 
    
    Parameters
    ----------
    weights : str
        path to .h5 file containing model weigts
    input_shape : tuple
        provide shape of model input
    config_file : str
        path to model config file (yaml)  
    aleatoric : boolean       
        set true if model has aleatoric uncertainty output head 
    epistemic : boolean
        set true if model has epistemic uncertainty output head
    with_evoaug : boolean
        set true if model was trained with EvoAug
    augment_list : list
        optional, if model was trained with EvoAug, provide list of augmentations used for training; required if with_evoaug=True 
    
    Returns
    -------
    keras.engine.functional.Functional 
        ResidualBind model loaded from weights
    '''
    config = yaml.safe_load(open(config_file, 'r'))
    model = None
    if with_evoaug:
        model = evoaug.RobustModel(lentiMPRA, input_shape=input_shape, augment_list=augment_list, max_augs_per_seq=1, hard_aug=True, config=config, aleatoric=aleatoric, epistemic=epistemic)
    else:
        model = lentiMPRA(input_shape, config, aleatoric=aleatoric, epistemic=epistemic)
    model.compile(optimizer=Adam(learning_rate=config['optim_lr']),
                  loss=config['loss_fxn'])
    model.load_weights(weights)
    return model

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    """
    Computes the Gaussian Negative Log-Likelihood (NLL) given mean and log variance.

    This function calculates the Gaussian NLL for a set of predictions, assuming 
    heteroscedastic uncertainty (variance that varies per data point) represented 
    by the log variance. This is often used in probabilistic regression.

    Parameters:
    - y (Tensor): The ground truth values.
    - mu (Tensor): The predicted mean values.
    - logvar (Tensor): The predicted log variance values.
    - reduce (bool): If True, computes the mean of the loss across all samples. 
                     If False, returns the loss per sample.

    Returns:
    - Tensor: The computed Gaussian NLL, either reduced to a mean value or 
              as a tensor of per-sample losses.
    """
    ax = list(range(1, len(y.shape)))
    
    log_likelihood = 0.5 * (
        -tf.exp(-logvar) * (mu - y)**2 
        - tf.math.log(2 * tf.constant(np.pi, dtype=logvar.dtype)) 
        - logvar
    )
    loss = tf.reduce_mean(-log_likelihood, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss


def EvidentialRegression(y_true, evidential_output):
    """
    Computes the loss for Evidential Regression based on Gaussian NLL.

    This function splits the evidential output into mean and log variance,
    then calculates the NLL using these parameters to measure the error between
    predictions and ground truth values with uncertainty estimates.

    Parameters:
    - y_true (Tensor): The ground truth values.
    - evidential_output (Tensor): The model output containing predicted mean 
                                  and log variance, concatenated along the last axis.

    Returns:
    - Tensor: The Gaussian NLL loss for evidential regression.
    """
    mu, logvar = tf.split(evidential_output, 2, axis=-1)
    loss_nll = Gaussian_NLL_logvar(y_true, mu, logvar)
    return loss_nll


def heteroscedastic_loss(y_true, y_pred):
    """
    Computes the heteroscedastic loss for probabilistic regression models.

    This function uses a predicted mean and standard deviation for each output 
    dimension and computes a likelihood-based loss. It accounts for the 
    heteroscedastic (input-dependent) noise in the predictions.

    Parameters:
    - y_true (Tensor): The ground truth values.
    - y_pred (Tensor): The model output containing predicted mean and 
                       standard deviation, concatenated along the last axis.

    Returns:
    - Tensor: The computed heteroscedastic loss as a scalar value.
    """
    n_outputs = y_true.shape[1]
    mu = y_pred[:, :n_outputs]
    std = y_pred[:, n_outputs:]
    
    return tf.reduce_mean(
        0.5 * (((y_true - mu)**2) / (std**2) + tf.math.log(2 * np.pi * std**2))
    )
