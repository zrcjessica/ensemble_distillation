import argparse
from os.path import join
import pickle
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import utils
from model_zoo import DeepSTARR
import plotting
import pandas as pd

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
    parser.add_argument("--lr", default=0.001,
                        help="fixed learning rate")
    parser.add_argument("--plot", action='store_true',
                        help="if set, save training plots")
    parser.add_argument("--downsample", default=None,
                        help="if set, downsample training data to this amount ([0,1))")
    parser.add_argument("--lr_decay", action="store_true",
                        help="if set, train with LR decay")
    args = parser.parse_args()
    return args

def main(args):

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)

    # downsample training data
    if args.downsample is not None:
        X_train, y_train =  utils.downsample(X_train, y_train, args.downsample)

    # create model 
    model = DeepSTARR(X_train[0].shape)

    # compile model
    model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')

    # define callbacks
    es_callback = EarlyStopping(patience=10, verbose=1, mode='min', restore_best_weights=True)
    callbacks_list = [es_callback]
    if args.lr_decay:
        lr_decay_callback = ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.2,
                                              patience=5,
                                              min_lr=1e-7,
                                              mode='min',
                                              verbose=1)
        callbacks_list.append(lr_decay_callback)

    # train model
    history = model.fit(X_train, y_train, 
                        batch_size=128, 
                        epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list)

    # evaluate model performance
    y_pred = model.predict(X_test)
    performance = utils.summarise_DeepSTARR_performance(y_pred, y_test)
    performance.to_csv(join(args.out, str(args.ix) + "_performance.csv"),
                       index=False)
    
    # plot loss curves and spearman correlation over training epochs and save 
    if args.plot:
        plotting.plot_loss(history, join(args.out, str(args.ix) + "_loss_curves.png"))

    # save model and history
    model.save(join(args.out, str(args.ix) + "_DeepSTARR.h5"))
    with open(join(args.out, str(args.ix) + "_historyDict"), 'wb') as pickle_fh:
        pickle.dump(history.history, pickle_fh)

if __name__ == "__main__":
    args = parse_args()
    main(args)