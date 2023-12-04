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
import wandb
from wandb.keras import WandbMetricsLogger
import yaml
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
    parser.add_argument("--lr", default=0.001,
                        help="fixed learning rate")
    parser.add_argument("--plot", action='store_true',
                        help="if set, save training plots")
    parser.add_argument("--downsample", default=1, type=float,
                        help="if set, downsample training data to this amount ([0,1])")
    # parser.add_argument("--early_stopping", action="store_true",
    #                     help="if set, train with early stopping")
    parser.add_argument("--lr_decay", action="store_true",
                        help="if set, train with LR decay")
    parser.add_argument("--project", type=str,
                        help='project name for wandb')
    parser.add_argument("--config", type=str,
                        help='path to wandb config (yaml)')
    parser.add_argument("--distill", type=str, default=None,
                        help='if provided, trains a distilled model using distilled training data')
    parser.add_argument("--k", type=int, default=1,
                        help='factor for adjusting number of parameters in hidden layers')
    args = parser.parse_args()
    return args

def main(args):

    # set up wandb for logging
    wandb.login()
    wandb.init(project=args.project, config=args.config)
    wandb.config['model_ix'] = args.ix

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data)

    # for training an ensemble distilled model
    if args.distill is not None:
        y_train = np.load(args.distill)
        wandb.config['distilled'] = True
        
    # downsample training data
    if args.downsample != wandb.config['downsample']:
        X_train, y_train = utils.downsample(X_train, y_train, args.downsample)
        wandb.config.update({'downsample':args.downsample}, allow_val_change=True)

    # adjust k in yaml 
    if args.k != wandb.config['k']:
        wandb.config.update({'k':args.k}, allow_val_change=True)

    # create model 
    model = DeepSTARR(X_train[0].shape, wandb.config)

    # compile model
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=wandb.config['loss_fxn'])
    
    # update lr in config if different value provided to input
    if args.lr != wandb.config['optim_lr']:
        wandb.config.update({'optim_lr':args.lr}, allow_val_change=True)

    # define callbacks
    callbacks_list = [WandbMetricsLogger()]
    if wandb.config['early_stopping']:
        es_callback = EarlyStopping(patience=wandb.config['es_patience'], verbose=1, mode='min', restore_best_weights=True)
        callbacks_list.append(es_callback)

    # if args.early_stopping:    
    #     es_callback = EarlyStopping(patience=10, verbose=1, mode='min', restore_best_weights=True)
    #     callbacks_list.append(es_callback)
    #     wandb.config.update({'early_stopping': True, 'es_patience': 10})

    if args.lr_decay:
        lr_decay_callback = ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.2,
                                              patience=5,
                                              min_lr=1e-7,
                                              mode='min',
                                              verbose=1)
        callbacks_list.append(lr_decay_callback)
        wandb.config.update({'lr_decay': True, 'lr_decay_patience': 3, 'lr_decay_factor': 0.2}, allow_val_change=True)

    # train model
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['epochs'],
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

    # save updated config as yaml
    with open(join(args.out, "config.yaml"), 'w') as f:
        yaml.dump(dict(wandb.config), f, allow_unicode=True, default_flow_style=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)