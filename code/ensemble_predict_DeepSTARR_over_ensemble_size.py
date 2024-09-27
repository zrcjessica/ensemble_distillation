import argparse
import utils
import plotting
import keras
from keras.models import load_model
from os.path import join
import numpy as np
import gc

'''
get the average of the predictions from all DeepSTARR models in an ensemble
applies trained ensemble models to test set
if --eval flag set, evaluates prediction performance on test set
if --distill flag set, averages predictions on train set from all DeepSTARR models in ensemble
if --std flag set, evaluate prediction performance on ensemble mean and std
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help='path directory storing trained ensemble of models')
    parser.add_argument("--n_mods", type=int, nargs="+",
                        help="number of models in ensemble (can provide multiple values to check)")
    parser.add_argument("--out", type=str,
                        help="output directory to save results")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    # parser.add_argument("--eval", action='store_true',
    #                     help='if set, evaluates average predictions and saves to file (ensemble_performance_avg.csv)')
    # parser.add_argument("--distill", action='store_true',
    #                     help='if set, writes average predictions on train set (X_train) to file (ensemble_avg_y_train.npy)')
    # parser.add_argument("--std", action='store_true',
    #                     help='if set, also evaluate performance on std predictions')
    # parser.add_argument("--downsample", type=float,
    #                     help='if set, downsample training data (only used if in distill mode)')
    parser.add_argument("--evoaug", action='store_true',
                        help='set if working with models trained w/ EvoAug')
    parser.add_argument("--config", default=None,
                        help='provide if --evoaug flag set; needed to load model from weights')
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)
    # if args.distill and args.downsample:
    #     # downsample training data
    #     rng = np.random.default_rng(1234)
    #     X_train, y_train = utils.downsample(X_train, y_train, rng, args.downsample)
    #     print(f'number of training samples after downsampling: {X_train.shape[0]}')

    # collect cumsum of predictions from each model in ensemble
    train_cumsum, test_cumsum = 0, 0

    max_mods = max(args.n_mods)
    # collect predictions on train, test, and val set for calculating ensemble std
    ensemble_preds_train = np.zeros((max_mods, X_train.shape[0], 2))
    ensemble_preds_test =  np.zeros((max_mods, X_test.shape[0], 2))
    ensemble_preds_val =  np.zeros((max_mods, X_val.shape[0], 2))
    
    # if no output directory is specified, use same as model_dir
    outdir = args.out
    if args.out is None:
        outdir = args.model_dir

    for i in range(max_mods):
        print(f'predicting with model {i+1}/{max_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model and predict 
        model = None
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
            #                                       predict_std=args.std)
            model = utils.load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_DeepSTARR_finetune.h5"), 
                                                  input_shape=X_train[0].shape, 
                                                  augment_list=augment_list, 
                                                  config_file=args.config, 
                                                  epistemic=False)
        else:
            model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        
        # train
        ensemble_preds_train[i,:,:] = model.predict(X_train)
        # test
        ensemble_preds_test[i,:,:] = model.predict(X_test)
        # val
        ensemble_preds_val[i,:,:] = model.predict(X_val)
        

    for n in args.n_mods:
        # calculate ensemble std 
        train_std = ensemble_preds_train[:n].std(axis=0)
        test_std = ensemble_preds_test[:n].std(axis=0)
        val_std = ensemble_preds_val[:n].std(axis=0)
        
        np.save(join(outdir, f"{n}mod_ensemble_std_train.npy"), train_std)
        np.save(join(outdir, f"{n}mod_ensemble_std_test.npy"), test_std)
        np.save(join(outdir, f"{n}mod_ensemble_std_val.npy"), val_std)
        
        # calculate ensemble avg on train set (for distillation)
        ensemble_avg_train = ensemble_preds_train[:n].mean(axis=0)
        # save average predictions on X_train to file and use for training distilled model
        np.save(join(outdir, f"{n}mod_ensemble_avg_y_train.npy"), ensemble_avg_train)
        
        # calculate ensemble avg on test set (for evaluating ensemble performance)
        ensemble_avg_test = ensemble_preds_test[:n].mean(axis=0)
        performance = utils.summarise_DeepSTARR_performance(ensemble_avg_test, y_test, std=False)    
        performance.to_csv(join(outdir, f"{n}mod_ensemble_performance_avg.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)