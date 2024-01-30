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
    parser.add_argument("--n_mods", type=int, 
                        help="number of models in ensemble")
    parser.add_argument("--out", type=str,
                        help="output directory to save results")
    parser.add_argument("--data", type=str,
                        help='h5 file containing train/val/test data')
    parser.add_argument("--eval", action='store_true',
                        help='if set, evaluates average predictions and saves to file (ensemble_performance_avg.csv)')
    parser.add_argument("--plot", action='store_true',
                        help='if set, generate scatterplots comparing predictions with ground truth')
    parser.add_argument("--distill", action='store_true',
                        help='if set, writes average predictions on train set (X_train) to file (ensemble_avg_y_train.npy)')
    parser.add_argument("--std", action='store_true',
                        help='if set, also evaluate performance on std predictions')
    # parser.add_argument("--set", default='test',
    #                     help='one of train/test/val; determine which set of data to make predictions on')
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data, std=args.std)
    # data_dict = utils.load_DeepSTARR_data(args.data, std=args.std, dict=True)

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

        # load model and predict 
        model = load_model(join(args.model_dir, str(i+1) + "_DeepSTARR.h5"))
        preds = 0
        if args.distill:
            # predict on train data to generate training data for distillation
            preds = model.predict(X_train)
        else:
            preds = model.predict(X_test)
        if args.plot:
            # plot pred vs. true
            plotting.prediction_scatterplot(preds, y_test,
                                            colnames=['Hk','Dev','Hk-std','Dev-std'][:(preds.shape[-1])],
                                            outfh=join(outdir, f'{i+1}_pred_scatterplot.png'))
        # preds = model.predict(data_dict[args.set]['X'])
        cumsum += preds
    
    # calculate average across ensemble predictions
    avg_pred = cumsum/args.n_mods 

    if args.eval:
        # evaluate performance + write to file
        performance = utils.summarise_DeepSTARR_performance(avg_pred, y_test, args.std)
        performance.to_csv(join(outdir, "ensemble_performance_avg.csv"), index=False)
        if args.plot:
            # plot average predictions against true values 
            plotting.prediction_scatterplot(avg_pred, y_test, 
                                            colnames=['Hk','Dev', 'Hk-std', 'Dev-std'][:(avg_pred.shape[-1])], 
                                            outfh=join(outdir, "avg_pred_scatterplot.png"))

    if args.distill:
        # save average predictions on X_train to file and use for training distilled model
        np.save(join(outdir, "ensemble_avg_y_train.npy"), avg_pred)

if __name__ == "__main__":
    args = parse_args()
    main(args)