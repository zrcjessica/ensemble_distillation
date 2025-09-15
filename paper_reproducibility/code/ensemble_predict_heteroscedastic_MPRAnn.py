import argparse
import utils
import plotting
import keras
from keras.models import Model
import tensorflow as tf
from os.path import join
import numpy as np
import gc
import yaml 
from model_zoo import MPRAnn_heteroscedastic

'''
get the average of the predictions from all MPRAnn models trained with heteroscedastic in an ensemble
applies trained ensemble models to test set
if --eval flag set, evaluates prediction performance on test set
if --distill flag set, averages predictions on train set from all MPRAnn models in ensemble
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
    parser.add_argument("--eval", action='store_true',
                        help='if set, evaluates average predictions and saves to file (ensemble_performance_avg.csv)')
    parser.add_argument("--plot", action='store_true',
                        help='if set, generate scatterplots comparing predictions with ground truth (only used in eval mode)')
    parser.add_argument("--distill", action='store_true',
                        help='if set, writes average predictions on train set (X_train) to file (ensemble_avg_y_train.npy)')
    # parser.add_argument("--aleatoric", action='store_true',
    #                     help='if set, predict aleatoric uncertainty')
    parser.add_argument("--downsample", type=float,
                        help='if set, downsample training data (only used if in distill mode)')
    parser.add_argument("--evoaug", action='store_true',
                        help='set if working with models trained w/ EvoAug')
    parser.add_argument("--config", default=None,
                        help='config file for model; needed to load model from weights')
    parser.add_argument("--celltype", type=str,
                    help='define celltype (K562/HepG2)')
    args = parser.parse_args()
    return args

# def heteroscedastic_loss(y_true, y_pred):
#     '''
#     heteroscedastic loss function 
#     '''
#     n_outputs = y_true.shape[1]
#     mu = y_pred[:, :n_outputs]
#     std = y_pred[:, n_outputs:]
    
#     return tf.reduce_mean(
#         0.5 * (((y_true - mu)**2) / (std**2) + tf.math.log(2*np.pi*std**2))
#     )

def load_model_from_weights(weights, input_shape, config_file, with_evoaug=False):
    '''
    load MPRAnn model trained with heteroscedastic regression from weights
    '''
    config = yaml.safe_load(open(config_file, 'r'))
    model = None
    if with_evoaug:
        import evoaug_tf
        from evoaug_tf import evoaug, augment
        augment_list = [
            augment.RandomDeletion(delete_min=0, delete_max=20),
            augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
            augment.RandomNoise(noise_mean=0, noise_std=0.2),
            augment.RandomMutation(mutate_frac=0.05)
            ]
        model = evoaug.RobustModel(MPRAnn_heteroscedastic, input_shape=input_shape, augment_list=augment_list, max_augs_per_seq=1, hard_aug=True)
    else:
        model = MPRAnn_heteroscedastic(input_shape)
    model.load_weights(weights)
    return model

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_lentiMPRA_data(args.data)

    if args.distill and args.downsample:
        # downsample training data
        rng = np.random.default_rng(1234)
        X_train, y_train = utils.downsample(X_train, y_train, rng, args.downsample)
        print(f'number of training samples after downsampling: {X_train.shape[0]}')

    # collect cumsum of predictions from each model in ensemble
    train_cumsum, test_cumsum = 0, 0

    # if no output directory is specified, use same as model_dir
    outdir = args.out if args.out else args.model_dir

    for i in range(args.n_mods):
        print(f'predicting with model {i+1}/{args.n_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model and predict 
        model = None
        if args.evoaug:
            model = load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_MPRAnn_heteroscedastic_finetune.h5"), 
                                                      input_shape=X_train[0].shape, 
                                                      config_file=args.config, 
                                                      with_evoaug=True)
        else:
            model = load_model_from_weights(weights=join(args.model_dir, f"{i+1}_MPRAnn_heteroscedastic.h5"),
                                            input_shape=X_train[0].shape,
                                            config_file=args.config)
        train_preds, test_preds = 0, 0
        if args.distill:
            # predict on train data to generate training data for distillation
            train_preds = model.predict(X_train)
            train_cumsum += train_preds
        if args.eval:
            test_preds = model.predict(X_test)
            if args.plot:
                # plot pred vs. true
                plotting.prediction_scatterplot(test_preds, y_test,
                                                colnames=[args.celltype, f'{args.celltype}-aleatoric'],
                                                outfh=join(outdir, f'{i+1}_pred_scatterplot.png'))
            test_cumsum += test_preds
    
    # calculate average across ensemble predictions
    avg_train_pred = train_cumsum/args.n_mods 
    avg_test_pred = test_cumsum/args.n_mods

    if args.eval:
        # evaluate performance + write to file
        if avg_test_pred.shape != y_test.shape:
            performance = utils.summarise_lentiMPRA_performance(avg_test_pred, 
                                                                np.expand_dims(y_test, axis=-1), 
                                                                args.celltype, 
                                                                aleatoric=True,
                                                                epistemic=False)
        else:
            performance = utils.summarise_lentiMPRA_performance(avg_test_pred, 
                                                                y_test, 
                                                                args.celltype, 
                                                                aleatoric=True,
                                                                epistemic=False)
        performance.to_csv(join(outdir, "ensemble_performance_avg.csv"), index=False)
        if args.plot:
            # plot average predictions against true values 
            plotting.prediction_scatterplot(avg_test_pred, y_test, 
                                            colnames=[args.celltype, f'{args.celltype}-aleatoric'], 
                                            outfh=join(outdir, "avg_pred_scatterplot.png"))

    if args.distill:
        # save average predictions on X_train to file and use for training distilled model
        np.save(join(outdir, f"ensemble_avg_y_train.npy"), avg_train_pred)

if __name__ == "__main__":
    args = parse_args()
    main(args)