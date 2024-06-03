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
import dynamic_aug 
from keras.models import load_model

'''
train distilled DeepSTARR models w/ mean+stdev prediction with dynamic augmentations 
--append determines whether augmentations are added to or replace original training data 
--aug determines what kind of augmentation: evoaug, random, mutagenesis 
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
    # parser.add_argument("--downsample", default=1, type=float,
    #                     help="if set, downsample training data to this amount ([0,1])")
    parser.add_argument("--lr_decay", action="store_true",
                        help="if set, train with LR decay")
    parser.add_argument("--project", type=str,
                        help='project name for wandb')
    parser.add_argument("--config", type=str,
                        help='path to wandb config (yaml)')
    # parser.add_argument("--distill", type=str, default=None,
    #                     help='if provided, trains a model using distilled training data')
    # parser.add_argument("--predict_std", action='store_true',
    #                     help='if set, predict ensemble stdev in addition to mean; distill flag must be set as well')
    parser.add_argument('--aug', type=str, default='evoaug',
                        help='define what type of augmentations to apply')
    parser.add_argument('--append', action='store_true',
                        help='if set, append augmentations to original data')
    parser.add_argument('--ensemble_dir', type=str,
                        help='path to dir where ensemble of models are stored; used to generate targets for augmented seqs')
    parser.add_argument('--ensemble_size', default=10,
                        help='number of models in ensemble to use for generating target labels')
    # parser.add_argument("--logvar", action='store_true',
    #                     help='if set, use logvar as target values instead of std (default)')
    args = parser.parse_args()
    return args

def eval_performance(model, X_test, y_test, outfh, logvar=False):
    y_pred = model.predict(X_test)
    if logvar:
        # convert logvar back to std for evaluation
        y_pred[:,2] = np.sqrt(np.exp(y_pred[:,2]))
        y_pred[:,3] = np.sqrt(np.exp(y_pred[:,3]))
    results = utils.summarise_DeepSTARR_performance(y_pred, y_test, std=True)
    results.to_csv(outfh, index=False)

def main(args):

    # set up wandb for logging
    wandb.login()
    wandb.init(project=args.project, config=args.config)
    wandb.config['model_ix'] = args.ix
    wandb.config['aug'] = args.aug
    wandb.config['append'] = args.append 
    wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config

    # load data from h5 (ensemble avg and std returned for y_train)
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data,  
                                                                               std=True)
    
    assert((X_train.shape[-1]==4) & (X_test.shape[-1]==4) & (X_val.shape[-1]==4))

    # # downsample training data
    # if args.downsample != wandb.config['downsample']:
    #     wandb.config.update({'downsample':args.downsample}, allow_val_change=True)
    #     if args.downsample<1:
    #         rng = np.random.default_rng(1234)
    #         X_train, y_train = utils.downsample(X_train, y_train, rng=rng, p=args.downsample, return_ix=False)
    
    # # use logvar instead of std as training targets 
    # if args.logvar:
    #     wandb.config.update({'std':False, 'logvar':True}, allow_val_change=True)
    #     y_train[:,2] = np.log(np.square(y_train[:,2]))
    #     y_train[:,3] = np.log(np.square(y_train[:,3]))
    #     y_val[:,2] = np.log(np.square(y_val[:,2]))
    #     y_val[:,3] = np.log(np.square(y_val[:,3]))

    # load ensemble of models
    ensemble = [load_model(join(args.ensemble_dir, f'{i}_DeepSTARR.h5')) for i in range(1,args.ensemble_size+1)]

    # create model
    model = None
    model = dynamic_aug.DynamicAugModel(DeepSTARR,
                                        input_shape=X_train[0].shape,
                                        aug=args.aug,
                                        append=args.append,
                                        ensemble=ensemble,
                                        config=wandb.config, 
                                        predict_std=True)

    # update lr in config if different value provided to input
    if args.lr != wandb.config['optim_lr']:
        wandb.config.update({'optim_lr':args.lr}, allow_val_change=True)

    # compile model
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=wandb.config['loss_fxn'])

    # define callbacks
    callbacks_list = [WandbMetricsLogger()]
    if wandb.config['early_stopping']:
        es_callback = EarlyStopping(patience=wandb.config['es_patience'], 
                                    verbose=1, 
                                    mode='min', 
                                    restore_best_weights=True)
        callbacks_list.append(es_callback)

    if args.lr_decay:
        # train w/ LR decay
        lr_decay_callback = ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.2,
                                              patience=5,
                                              min_lr=1e-7,
                                              mode='min',
                                              verbose=1)
        callbacks_list.append(lr_decay_callback)
        wandb.config.update({'lr_decay': True, 
                             'lr_decay_patience': 3, 
                             'lr_decay_factor': 0.2}, allow_val_change=True)

    # train model
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
    
    # save model weights
    if args.append:
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}_append.h5")
        model.save_weights(save_path)
        # save history
        with open(join(args.out, f"{args.ix}_{args.aug}_append_historyDict"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        # evaluate best model (and save)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_append_performance.csv'))

        # plot loss curves and spearman correlation over training epochs and save 
        if args.plot:
            plotting.plot_loss(history, join(args.out, f"{args.ix}_{args.aug}_append_loss_curves.png"))
    else:
        # save model weights
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}.h5")
        model.save_weights(save_path)
        # save history
        with open(join(args.out, f"{args.ix}_{args.aug}_historyDict"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        # evaluate best model (and save)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_performance.csv'))

        # plot loss curves and spearman correlation over training epochs and save 
        if args.plot:
            plotting.plot_loss(history, join(args.out, f"{args.ix}_{args.aug}_loss_curves.png"))
    
    # save updated config as yaml
    with open(join(args.out, "config.yaml"), 'w') as f:
        yaml.dump(dict(wandb.config), f, allow_unicode=True, default_flow_style=False)

    wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)