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
    parser.add_argument("--evoaug", action='store_true',
                        help='set when using models trained w/ EvoAug')
    parser.add_argument("--config", default=None,
                        help='provide if --evoaug flag set; needed to load model from weights')
    parser.add_argument("--downsample", default=1, type=float,
                        help="if set, downsample training data to this amount ([0,1])")
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)

    # downsample training data
    if args.downsample < 1:
        rng = np.random.default_rng(1234)
        X_train, y_train = utils.downsample(X_train, y_train, rng, args.downsample)

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

        # load model 
        if args.evoaug:
            import evoaug_tf
            from evoaug_tf import evoaug, augment
            # augment_list = [
            #     augment.RandomInsertionBatch(insert_min=0, insert_max=20),
            #     augment.RandomDeletion(delete_min=0, delete_max=30),
            #     augment.RandomTranslocationBatch(shift_min=0, shift_max=20)
            # ]   
            augment_list = [
                augment.RandomDeletion(delete_min=0, delete_max=20),
                augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
                augment.RandomNoise(noise_mean=0, noise_std=0.2),
                augment.RandomMutation(mutate_frac=0.05)
            ]
            # model = utils.load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_DeepSTARR_finetune.h5"), 
            #                                       input_shape=X_train[0].shape, 
            #                                       augment_list=augment_list, 
            #                                       config_file=args.config, 
            #                                       predict_std=False)
            model = utils.load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_DeepSTARR_finetune.h5"), 
                                                  input_shape=X_train[0].shape, 
                                                  augment_list=augment_list, 
                                                  config_file=args.config, 
                                                  epistemic=False)
        else:
            model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))

        # # load model and predict on test data
        # model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))

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
    if args.downsample<1:
        np.save(join(outdir, f"downsample{args.downsample}_ensemble_std_train.npy"), train_std)
        np.save(join(outdir, f"downsample{args.downsample}_ensemble_std_test.npy"), test_std)
        np.save(join(outdir, f"downsample{args.downsample}_ensemble_std_val.npy"), val_std)
    else:
        np.save(join(outdir, "ensemble_std_train.npy"), train_std)
        np.save(join(outdir, "ensemble_std_test.npy"), test_std)
        np.save(join(outdir, "ensemble_std_val.npy"), val_std)

if __name__ == "__main__":
    args = parse_args()
    main(args)