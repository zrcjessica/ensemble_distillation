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
train distilled DeepSTARR models w/ stdev prediction 
--downsample flag allows models to be trained with a subset of training data
--evoaug flag determines whether model will be trained w/ evoaug augmentations
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
    parser.add_argument("--first_activation", default='relu',
                        help='if provided, defines the first layer activation function')
    # parser.add_argument("--early_stopping", action="store_true",
    #                     help="if set, train with early stopping")
    parser.add_argument("--lr_decay", action="store_true",
                        help="if set, train with LR decay")
    parser.add_argument("--project", type=str,
                        help='project name for wandb')
    parser.add_argument("--config", type=str,
                        help='path to wandb config (yaml)')
    # parser.add_argument("--distill", type=str, default=None,
    #                     help='if provided, trains a   model using distilled training data')
    # parser.add_argument("--distill", action='store_true',
    #                     help='if set, train distilled DeepSTARR models')
    parser.add_argument("--k", type=int, default=1,
                        help='factor for adjusting number of parameters in hidden layers')
    # parser.add_argument("--predict_std", action='store_true',
    #                     help='if set, predict ensemble stdev in addition to mean; distill flag must be set as well')
    parser.add_argument("--evoaug", action='store_true',
                        help='if set, train models with evoaug')
    args = parser.parse_args()
    return args

def eval_performance(model, X_test, y_test, args, outfh):
    y_pred = model.predict(X_test)
    aug_results = utils.summarise_DeepSTARR_performance(y_pred, y_test, std=True)
    aug_results.to_csv(outfh, index=False)

def main(args):

    # set up wandb for logging
    wandb.login()
    wandb.init(project=args.project, config=args.config)
    wandb.config['model_ix'] = args.ix
    wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data,  
                                                                               std=True)
    
    assert((X_train.shape[-1]==4) & (X_test.shape[-1]==4) & (X_val.shape[-1]==4))

    # downsample training data
    if args.downsample != wandb.config['downsample']:
        wandb.config.update({'downsample':args.downsample}, allow_val_change=True)
        if args.downsample<1:
            X_train, y_train = utils.downsample(X_train, y_train, args.downsample)
        
    # # for training an ensemble distilled model
    # if args.distill is not None:
    #     wandb.config.update({'distill':True}, allow_val_change=True)
    #     y_train = np.load(args.distill)
    #     assert(X_train.shape[0]==y_train.shape[0])
    #     if args.predict_std != wandb.config['std']:
    #         # predict std
    #         wandb.config.update({'std':args.predict_std}, allow_val_change=True)

    # adjust k in yaml 
    if args.k != wandb.config['k']:
        wandb.config.update({'k':args.k}, allow_val_change=True)

    # adjust first_layer_activation in yaml
    if args.first_activation != wandb.config['first_activation']:
        wandb.config.update({'first_activation':args.first_activation}, allow_val_change=True)

    # create model
    model = None
    augment_list = None
    if args.evoaug:
        # for training w/ evoaug
        import evoaug_tf
        from evoaug_tf import evoaug, augment
        augment_list = [
            # augment.RandomRC(rc_prob=0.5),
            augment.RandomInsertionBatch(insert_min=0, insert_max=20),
            augment.RandomDeletion(delete_min=0, delete_max=30),
            augment.RandomTranslocationBatch(shift_min=0, shift_max=20)
            # augment.RandomNoise(noise_mean=0, noise_std=0.3),
            # augment.RandomMutation(mutate_frac=0.05)
        ]   
        wandb.config.update({'evoaug':True}, allow_val_change=True)
        wandb.config['finetune'] = False
        model = evoaug.RobustModel(DeepSTARR, input_shape=X_train[0].shape, augment_list=augment_list, max_augs_per_seq=1, hard_aug=True, config=wandb.config, predict_std=True)
    else:
        # training w/o evoaug
        model = DeepSTARR(X_train[0].shape, config=wandb.config, predict_std=True)

    # update lr in config if different value provided to input
    if args.lr != wandb.config['optim_lr']:
        wandb.config.update({'optim_lr':args.lr}, allow_val_change=True)

    # compile model
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=wandb.config['loss_fxn'])

    # define callbacks
    callbacks_list = [WandbMetricsLogger()]
    if wandb.config['early_stopping']:
        es_callback = EarlyStopping(patience=wandb.config['es_patience'], verbose=1, mode='min', restore_best_weights=True)
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
        wandb.config.update({'lr_decay': True, 'lr_decay_patience': 3, 'lr_decay_factor': 0.2}, allow_val_change=True)

    # train model
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
    if args.evoaug:
        # save weights
        save_path = join(args.out, f"{args.ix}_DeepSTARR_aug_weights.h5")
        model.save_weights(save_path)
        # save history
        with open(join(args.out, str(args.ix) + "_historyDict_aug"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        
        # evaluate best model (and save)
        eval_performance(model, X_test, y_test, args, join(args.out, f'{args.ix}_performance_aug.csv'))

        ### fine tune model (w/o augmentations)
        wandb.config.update({'finetune':True}, allow_val_change=True)
        finetune_epochs = 10
        wandb.config['finetune_epochs']=finetune_epochs
        model = evoaug.RobustModel(DeepSTARR, input_shape=X_train[0].shape, augment_list=augment_list, max_augs_per_seq=2, hard_aug=True, config=wandb.config, predict_std=True)
        model.compile(optimizer=Adam(learning_rate=args.lr), loss=wandb.config['loss_fxn'])
        model.load_weights(save_path)
        model.finetune_mode()
        # train
        history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=finetune_epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
        # save model and history
        model.save_weights(join(args.out, f"{args.ix}_DeepSTARR_finetune.h5"))
        with open(join(args.out, f"{args.ix}_historyDict_finetune"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        # evaluate model performance
        eval_performance(model, X_test, y_test, args, join(args.out, f'{args.ix}_performance_finetune.csv'))
    else:
        # evaluate model performance
        eval_performance(model, X_test, y_test, args, join(args.out, str(args.ix) + "_performance.csv"))
    
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