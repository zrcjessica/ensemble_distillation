import utils
import argparse
from os.path import join
import keras
from keras.models import load_model
import numpy as np
import gc

'''
get average of ensemble predictions for X_train to generate new y_train
new y_train will be used to train an ensemble distilled model
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help='path directory storing trained ensemble of models')
    parser.add_argument("--n_mods", type=int, 
                        help="number of models in ensemble")
    parser.add_argument("--out", type=str,
                        help="output directory to save results; defaults to model_dir if not provided")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)

    # collect cumsum of predictions from each model in ensemble
    cumsum = 0

    # if no output directory is specified, use same as model_dir
    outdir = args.out
    if args.out is None:
        outdir = args.model_dir

    # iterate through models 
    for i in range(args.n_mods):
        print(f'predicting with model {i+1}/{args.n_mods}')
        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model and predict on test data
        model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        cumsum += model.predict(X_train)
    
    # calculate average across ensemble predictions
    avg_pred = cumsum/args.n_mods 

    # save average of predictions to file
    np.save(join(outdir, "ensemble_mean_y_train.npy"), avg_pred)

if __name__ == "__main__":
    args = parse_args()
    main(args)