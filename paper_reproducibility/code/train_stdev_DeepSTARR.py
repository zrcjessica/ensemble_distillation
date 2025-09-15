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
from wandb.integration.keras import WandbMetricsLogger
import yaml
import numpy as np

'''
train distilled DeepSTARR model w/ ensemble mean+std 
--downsample flag allows models to be trained with a subset of training data
--evoaug flag indicates model will be trained w/ evoaug augmentations
--logvar flag trains w/ logvar targets but evaluates performance w/ std
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
    parser.add_argument("--k", type=int, default=1,
                        help='factor for adjusting number of parameters in hidden layers')
    parser.add_argument("--evoaug", action='store_true',
                        help='if set, train models with evoaug')
    parser.add_argument("--logvar", action='store_true',
                        help='if set, use logvar as target values instead of std (default)')
    parser.add_argument("--nmods", type=int, action='store', default=10, 
                         help='size of teacher ensemble')
    args = parser.parse_args()
    return args

def eval_performance(model, X_test, y_test, outfh, logvar=False):
    """Evaluates model performance on test sequences and saves to .csv

    Parameters
    ----------
    model : keras.engine.functional.Functional
        Model to evaluate performance on 
    X_test : np.array
        Test sequences 
    y_test : np.array
        Labels for test sequences 
    outfh : str
        Output filehandle 
    """
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
    run = wandb.init(project=args.project, config=args.config, reinit=True)
    wandb.config['model_ix'] = args.ix
    wandb.config['n_teachers'] = args.nmods
    print(f'training distilled model from teacher ensemble of size {args.nmods} models')
    wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config

    # load data from h5 (ensemble avg and std returned for y_train)
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data, std=True)
    
    assert((X_train.shape[-1]==4) & (X_test.shape[-1]==4) & (X_val.shape[-1]==4))

    # downsample training data
    if args.downsample != wandb.config['downsample']:
        wandb.config.update({'downsample':args.downsample}, allow_val_change=True)
    
    # use logvar instead of std as training targets 
    if args.logvar:
        wandb.config.update({'std':False, 'logvar':True}, allow_val_change=True)
        y_train[:,2] = np.log(np.square(y_train[:,2]))
        y_train[:,3] = np.log(np.square(y_train[:,3]))
        y_val[:,2] = np.log(np.square(y_val[:,2]))
        y_val[:,3] = np.log(np.square(y_val[:,3]))

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
                augment.RandomDeletion(delete_min=0, delete_max=20),
                augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
                augment.RandomNoise(noise_mean=0, noise_std=0.2),
                augment.RandomMutation(mutate_frac=0.05)
        ]  
        wandb.config.update({'evoaug':True}, allow_val_change=True)
        wandb.config['finetune'] = False
        model = evoaug.RobustModel(DeepSTARR, input_shape=X_train[0].shape, 
                                   augment_list=augment_list, 
                                   max_augs_per_seq=1, 
                                   hard_aug=True, 
                                   config=wandb.config, 
                                   epistemic=True)
    else:
        # training w/o evoaug
        model = DeepSTARR(X_train[0].shape, config=wandb.config, epistemic=True)

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
        wandb.config.update({'lr_decay': True, 
                             'lr_decay_patience': 3, 
                             'lr_decay_factor': 0.2}, 
                             allow_val_change=True)

    # train model
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 

    if args.evoaug:
        run.finish()

        # save weights
        save_path = join(args.out, f"{args.ix}_DeepSTARR_aug_weights.h5")
        model.save_weights(save_path)

        # save history
        with open(join(args.out, str(args.ix) + "_historyDict_aug"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        
        # evaluate best model (and save)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_performance_aug.csv'), args.logvar)

        ### fine tune model (w/o augmentations)
        run = wandb.init(project=args.project, config=args.config, reinit=True)
        wandb.config['model_ix'] = args.ix
        wandb.config['finetune'] = True
        wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config
        if args.lr_decay:
            wandb.config.update({'lr_decay': True, 
                                'lr_decay_patience': 3, 
                                'lr_decay_factor': 0.2}, 
                                allow_val_change=True)
        # wandb.config.update({'finetune':True}, allow_val_change=True)
        finetune_epochs = 10
        wandb.config['finetune_epochs'] = finetune_epochs
        wandb.config['finetune_lr'] = 0.0001
        finetune_optimizer = Adam(learning_rate=0.0001)
        model = evoaug.RobustModel(DeepSTARR, input_shape=X_train[0].shape, 
                                   augment_list=augment_list, 
                                   max_augs_per_seq=2, 
                                   hard_aug=True, 
                                   config=wandb.config, 
                                   epistemic=True)
        model.compile(optimizer=finetune_optimizer, loss=wandb.config['loss_fxn'])
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
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_performance_finetune.csv'), args.logvar)
    else:
        # evaluate model performance
        eval_performance(model, X_test, y_test, join(args.out, str(args.ix) + "_performance.csv"), args.logvar)
    
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
    
    run.finish()
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)