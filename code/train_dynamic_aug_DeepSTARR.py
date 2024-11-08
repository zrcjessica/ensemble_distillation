import argparse
from os.path import join
import pickle
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import utils
import h5py
from model_zoo import DeepSTARR
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger
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
    parser.add_argument('--aug', type=str, default='evoaug',
                        help='define what type of augmentations to apply (mutagenesis, evoaug, random)')
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
    """Evaluate model performance on test data and save results."""
    y_pred = model.predict(X_test)
    if logvar:
        # convert logvar back to std for evaluation
        y_pred[:,2] = np.sqrt(np.exp(y_pred[:,2]))
        y_pred[:,3] = np.sqrt(np.exp(y_pred[:,3]))
    results = utils.summarise_DeepSTARR_performance(y_pred, y_test, std=True)
    results.to_csv(outfh, index=False)

def main(args):
    # Set up wandb for experiment tracking
    wandb.login()
    run = wandb.init(project=args.project, config=args.config, reinit=True)
    wandb.config['model_ix'] = args.ix
    wandb.config['aug'] = args.aug
    wandb.config['append'] = args.append 
    wandb.config['finetune'] = False
    wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config

    # Load data from HDF5 file (ensemble avg and std returned for y_train)
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_DeepSTARR_data(args.data,  
                                                                               std=True)
    assert((y_train.shape[-1]==4) & (y_test.shape[-1]==4) & (y_val.shape[-1]==4))

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

    # Load ensemble of pre-trained models
    ensemble = [load_model(join(args.ensemble_dir, f'{i}_DeepSTARR.h5')) for i in range(1,args.ensemble_size+1)]

    # Initialize model with dynamic augmentations
    model = dynamic_aug.DynamicAugModel(DeepSTARR,
                                        input_shape=X_train[0].shape,
                                        aug=args.aug,
                                        append=args.append,
                                        ensemble=ensemble,
                                        config=wandb.config, 
                                        epistemic=True)

    # Update learning rate in config if a new value is provided
    if args.lr != wandb.config['optim_lr']:
        wandb.config.update({'optim_lr':args.lr}, allow_val_change=True)

    # Compile model with Adam optimizer and specified loss function
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=wandb.config['loss_fxn'])

    # Define callbacks for training
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

    # Train model with augmentations
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
    
    run.finish()

    # Save model weights, history, and performance evaluation
    if args.append:
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}_append_aug.h5")
        model.save_weights(save_path)
        with open(join(args.out, f"{args.ix}_{args.aug}_append_historyDict_aug"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_append_performance_aug.csv'))
    else:
        # Non-append version of saving
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}_aug.h5")
        model.save_weights(save_path)
        with open(join(args.out, f"{args.ix}_{args.aug}_historyDict_aug"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_performance_aug.csv'))
    
    # Fine-tune the model without augmentations
    run = wandb.init(project=args.project, config=args.config, reinit=True)
    wandb.config['model_ix'] = args.ix
    wandb.config['aug'] = args.aug
    wandb.config['append'] = args.append 
    wandb.config['finetune'] = True
    wandb.config.update({'distill':True, 'std':True}, allow_val_change=True) # update config
    wandb.config.update({'lr_decay': True, 
                         'lr_decay_patience': 3, 
                         'lr_decay_factor': 0.2}, 
                         allow_val_change=True)
    finetune_epochs = 100
    wandb.config['finetune_epochs'] = finetune_epochs
    wandb.config['finetune_lr'] = 0.001
    finetune_optimizer = keras.optimizers.Adam(learning_rate=wandb.config['finetune_lr'])
    model = dynamic_aug.DynamicAugModel(DeepSTARR,
                                        input_shape=X_train[0].shape,
                                        aug=None,
                                        append=False,
                                        config=wandb.config, 
                                        epistemic=True)
    model.compile(optimizer=finetune_optimizer, loss=wandb.config['loss_fxn'])
    model.load_weights(save_path)

    # Fine-tune training
    history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=wandb.config['finetune_epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
    

     # Save fine-tuned model and performance evaluation
    if args.append:
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}_append_finetune.h5")
        model.save_weights(save_path)
        with open(join(args.out, f"{args.ix}_{args.aug}_append_historyDict_finetune"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_append_performance_finetune.csv'))
    else:
        save_path = join(args.out, f"{args.ix}_DeepSTARR_{args.aug}_finetune.h5")
        model.save_weights(save_path)
        with open(join(args.out, f"{args.ix}_{args.aug}_historyDict_finetune"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        eval_performance(model, X_test, y_test, join(args.out, f'{args.ix}_{args.aug}_performance_finetune.csv'))

    # save updated model config as yaml
    with open(join(args.out, "config.yaml"), 'w') as f:
        yaml.dump(dict(wandb.config), f, allow_unicode=True, default_flow_style=False)

    run.finish()
    wandb.finish()

    
if __name__ == "__main__":
    args = parse_args()
    main(args)