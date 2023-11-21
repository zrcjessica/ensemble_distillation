import argparse
import utils
import keras
from keras.models import load_model
from os.path import join
import numpy as np
import gc

'''
get the average of the predictions from all DeepSTARR models in an ensemble
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help='path directory storing trained ensemble of models')
    parser.add_argument("--n_mods", type=int, 
                        help="number of models in ensemble")
    parser.add_argument("--out", type=str,
                        help="output directory to save results")
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

    for i in range(args.n_mods):
        print(f'predicting with model {i+1}/{args.n_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model and predict on test data
        model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        cumsum += model.predict(X_test)
    
    # calculate average across ensemble predictions
    avg_pred = cumsum/args.n_mods 

    # evaluate performance + write to file
    performance = summarise_DeepSTARR_performance(avg_pred, y_test)
    performance.to_csv(join(outdir, "ensemble_performance.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)