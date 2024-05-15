import argparse
import utils
import plotting
import keras
from keras.models import load_model
from os.path import join
import numpy as np
import gc

'''
get the average of the predictions from all lentiMPRA models in an ensemble
applies trained ensemble models to test set
if --eval flag set, evaluates prediction performance on test set
if --distill flag set, averages predictions on train set from all lentiMPRA models in ensemble
if --std flag set, evaluate prediction performance on ensemble mean and std
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
    parser.add_argument("--std", action='store_true',
                        help='if set, also evaluate performance on std predictions')
    parser.add_argument("--downsample", type=float,
                        help='if set, downsample training data (only used if in distill mode)')
    # parser.add_argument("--set", default='test',
    #                     help='one of train/test/val; determine which set of data to make predictions on')
    parser.add_argument("--evoaug", action='store_true',
                        help='set if working with models trained w/ EvoAug')
    parser.add_argument("--config", default=None,
                        help='provide if --evoaug flag set; needed to load model from weights')
    parser.add_argument("--celltype", type=str,
                    help='define celltype (K562/HepG2)')
    args = parser.parse_args()
    return args

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
    outdir = args.out
    if args.out is None:
        outdir = args.model_dir

    for i in range(args.n_mods):
        print(f'predicting with model {i+1}/{args.n_mods}')

        # clear history
        keras.backend.clear_session()
        gc.collect()

        # load model and predict 
        model = None
        if args.evoaug:
            assert(args.config is not None)
            import evoaug_tf
            from evoaug_tf import evoaug, augment
            augment_list = [
                augment.RandomInsertionBatch(insert_min=0, insert_max=20),
                augment.RandomDeletion(delete_min=0, delete_max=30),
                augment.RandomTranslocationBatch(shift_min=0, shift_max=20)
            ]   
            model = utils.load_model_from_weights(weights=join(args.model_dir, str(i+1) + "_lentiMPRA_finetune.h5"), 
                                                  input_shape=X_train[0].shape, 
                                                  augment_list=augment_list, 
                                                  config_file=args.config, 
                                                  predict_std=args.std)
        else:
            model = load_model(join(args.model_dir, str(i+1) + "_lentiMPRA.h5"))
        train_preds, test_preds = 0, 0
        if args.distill:
            # predict on train data to generate training data for distillation
            train_preds = model.predict(X_train)
            train_cumsum += train_preds
        if args.eval:
            test_preds = model.predict(X_test)
            if args.plot:
                # plot pred vs. true
                plotting.prediction_scatterplot(np.squeeze(test_preds), y_test,
                                                colnames=[args.celltype, f'{args.celltype}-std'][:(test_preds.shape[-1])],
                                                outfh=join(outdir, f'{i+1}_pred_scatterplot.png'))
            test_cumsum += test_preds
    
    # calculate average across ensemble predictions
    avg_train_pred = train_cumsum/args.n_mods 
    avg_test_pred = test_cumsum/args.n_mods

    if args.eval:
        # evaluate performance + write to file
        if avg_test_pred.shape != y_test.shape:
            performance = utils.summarise_lentiMPRA_performance(avg_test_pred, np.expand_dims(y_test, axis=-1), args.celltype, args.std)
        else:
            performance = utils.summarise_lentiMPRA_performance(avg_test_pred, y_test, args.celltype, args.std)
        performance.to_csv(join(outdir, "ensemble_performance_avg.csv"), index=False)
        if args.plot:
            # plot average predictions against true values 
            plotting.prediction_scatterplot(np.squeeze(avg_test_pred), y_test, 
                                            colnames=[args.celltype, f'{args.celltype}-std'][:(test_preds.shape[-1])], 
                                            outfh=join(outdir, "avg_pred_scatterplot.png"))

    if args.distill:
        # save average predictions on X_train to file and use for training distilled model
        np.save(join(outdir, "ensemble_avg_y_train.npy"), avg_train_pred)

if __name__ == "__main__":
    args = parse_args()
    main(args)