import argparse
from os.path import join
# import pickle
# import keras
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import utils
# from model_zoo import DeepSTARR
# import plotting
# import pandas as pd
# import yaml
import numpy as np

'''
train an ensemble of DeepSTARR models
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ix", type=int, 
                        help="ix of model in ensemble, appended to output file names")
    parser.add_argument("--out", type=str,
                        help="output directory to save model and plots")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    parser.add_argument("--downsample", default=1, type=float,
                        help="if set, downsample training data to this amount ([0,1])")
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data, 
                                                                               ensemble=False, 
                                                                               std=False)
    
    # downsample training data
    ix = utils.downsample_eval(X_train, y_train, args.downsample)

    # save to file to check later
    np.save(join(args.out, f"{args.ix}_downsample_ix.npy"), ix)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)