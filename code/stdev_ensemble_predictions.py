import argparse
import utils
import keras
from keras.models import load_model
from os.path import join
import numpy as np
import gc

'''
calculate standard deviation of ensemble predictions for train/test/val data
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
                        help='h5 file containing original train/val/test data')
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)

    # empty matrix to collect predictions 
    ensemble_preds_train = np.zeros((args.n_mods, X_train.shape[0], 2))
    ensemble_preds_test =  np.zeros((args.n_mods, X_test.shape[0], 2))
    ensemble_preds_val =  np.zeros((args.n_mods, X_val.shape[0], 2))

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

        # train
        ensemble_preds_train[i,:,:] = model.predict(X_train)
        # test
        ensemble_preds_test[i,:,:] = model.predict(X_test)
        # val
        ensemble_preds_val[i,:,:] = model.predict(X_val)

    ### calculate standard deviation
    # train
    train_std = np.std(ensemble_preds_train, axis=0)
    # test
    test_std = np.std(ensemble_preds_test, axis=0)
    # val
    val_std = np.std(ensemble_preds_val, axis=0)
    
    ### write to files
    np.save(join(outdir, "ensemble_std_train.npy"), train_std)
    np.save(join(outdir, "ensemble_std_test.npy"), test_std)
    np.save(join(outdir, "ensemble_std_val.npy"), val_std)

if __name__ == "__main__":
    args = parse_args()
    main(args)