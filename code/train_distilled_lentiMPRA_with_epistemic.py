import argparse
from os.path import join
import pickle
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import utils
from model_zoo import lentiMPRA
import plotting
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger
import yaml
import numpy as np

'''
train distilled ResidualBind models w/ mean+aleatoric+epistemic predictions 
assumes that HDF5 file provided to --data contains ensemble avg/std 
for training downsampled models, make sure that h5 file provided to --data corresponds to downsample proportion
--evoaug flag determines whether model will be trained w/ EvoAug 
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
    parser.add_argument("--downsample", default=1, type=float,
                        help="if set, downsample training data to this amount ([0,1])")
    parser.add_argument("--lr_decay", action="store_true",
                        help="if set, train with LR decay")
    parser.add_argument("--project", type=str,
                        help='project name for wandb')
    parser.add_argument("--config", type=str,
                        help='path to wandb config (yaml)')
    parser.add_argument("--evoaug", action='store_true',
                        help='if set, train models with evoaug')
    parser.add_argument("--celltype", type=str,
                        help='define celltype (K562/HepG2)')
    parser.add_argument("--logvar", action='store_true',
                        help='if set, predict uncertainties as logvar instead of std')
    args = parser.parse_args()
    return args

def eval_performance(model, X_test, y_test, outfh, celltype, logvar=False):
    """
    Evaluate the performance of distilled model a test sequences and saves results to a CSV file.

    Parameters:
    -----------
    model : keras.Model
        Distilled ResidualBind model
    X_test : np.ndarray
        One-hot encoded test sequences from lentiMPRA 
    y_test : np.ndarray
        Target labels corresponding to test sequences; should reflect ensemble average/std
    outfh : str
        Output file path to save evaluation results in CSV format.
    celltype : str
        The cell type associated with the dataset ('K562' or 'HepG2'), used for labeling.
    logvar : bool, optional
        If True, the model predicts uncertainty as log-variance, which is converted back to standard deviation for evaluation.
        Default is False.

    Returns:
    --------
    None
        Results are saved to a CSV file at the specified output path.

    Notes:
    ------
    - Assumes that the model predictions have three output dimensions (e.g., point estimate, aleatoric, and epistemic uncertainties).
    - Converts log-variance uncertainty estimates to standard deviation if `logvar` is True.
    - Uses `utils.summarise_lentiMPRA_performance` to summarize and save the evaluation metrics, which considers both aleatoric and epistemic uncertainties.
    """
    y_pred = model.predict(X_test)
    if logvar:
        # convert logvar back to std for evaluation
        y_pred[:,1:] = np.sqrt(np.exp(y_pred[:,1:]))
    results = None
    assert(y_pred.shape[-1]==3)
    if y_pred.shape != y_test.shape:
        results = utils.summarise_lentiMPRA_performance(y_pred, np.expand_dims(y_test, axis=-1), celltype, aleatoric=True, epistemic=True)
    else:
        results = utils.summarise_lentiMPRA_performance(y_pred, y_test, celltype, aleatoric=True, epistemic=True)
    results.to_csv(outfh, index=False)

def main(args):

    # set up wandb for logging
    wandb.login()
    wandb.init(project=args.project, config=args.config)
    wandb.config['model_ix'] = args.ix
    wandb.config['celltype'] = args.celltype
    wandb.config.update({'distill':True, 'aleatoric':True, 'epistemic':True}, allow_val_change=True)

    # load data from h5
    X_train, y_train, X_test, y_test, X_val, y_val = utils.load_lentiMPRA_data(file=args.data, epistemic=True)

    assert(y_train.shape[-1]==3)
    assert(y_test.shape[-1]==3)
    assert(y_val.shape[-1]==3)
    
    # downsample training data
    if args.downsample != wandb.config['downsample']:
        # if args.downsample is True, we assumed user has passed .h5 corresponding to downsampled data
        # no further downsampling applied
        wandb.config.update({'downsample':args.downsample}, allow_val_change=True)

    # use logvar instead of std as training targets for epistemic head (last column)
    if args.logvar:
        wandb.config.update({'std':False, 'logvar':True}, allow_val_change=True)
        y_train[:,-1] = np.log(np.square(y_train[:,1:]))
        y_val[:,-1] = np.log(np.square(y_val[:,1:]))

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
        model = evoaug.RobustModel(lentiMPRA, 
                                   input_shape=X_train[0].shape, 
                                   augment_list=augment_list, 
                                   max_augs_per_seq=1, 
                                   hard_aug=True, 
                                   config=wandb.config, 
                                   aleatoric=True,
                                   epistemic=True)
    else:
        # training w/o evoaug
        model = lentiMPRA(X_train[0].shape, wandb.config, aleatoric=True, epistemic=True)

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
        # save_path = join(args.out, f"{args.ix}_lentiMPRA_aug_weights.h5")
        save_path = join(args.out, f"{args.ix}_ResidualBind_aug_weights.h5")
        model.save_weights(save_path)
        # save history
        with open(join(args.out, str(args.ix) + "_historyDict_aug"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        
        # evaluate best model (and save)
        eval_performance(model, X_test, y_test, 
                         join(args.out, f'{args.ix}_performance_aug.csv'), 
                         args.celltype, 
                         logvar=args.logvar)

        ### fine tune model (w/o augmentations)
        wandb.config.update({'finetune':True}, allow_val_change=True)
        finetune_epochs = 10
        wandb.config['finetune_epochs']=finetune_epochs
        wandb.config['finetune_lr'] = 0.0001
        model = evoaug.RobustModel(lentiMPRA, input_shape=X_train[0].shape, 
                                   augment_list=augment_list, 
                                   max_augs_per_seq=2, hard_aug=True, 
                                   config=wandb.config, 
                                   aleatoric=True, epistemic=True)
        model.compile(optimizer=Adam(learning_rate=wandb.config['finetune_lr']), loss=wandb.config['loss_fxn'])
        model.load_weights(save_path)
        model.finetune_mode()
        # train
        history = model.fit(X_train, y_train, 
                        batch_size=wandb.config['batch_size'], 
                        epochs=finetune_epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list) 
        # save model and history
        # model.save_weights(join(args.out, f"{args.ix}_lentiMPRA_finetune.h5"))
        model.save_weights(join(args.out, f"{args.ix}_ResidualBind_finetune.h5"))
        with open(join(args.out, f"{args.ix}_historyDict_finetune"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)
        # evaluate model performance
        eval_performance(model, X_test, y_test, 
                         join(args.out, f'{args.ix}_performance_finetune.csv'), 
                         args.celltype, logvar=args.logvar)
    else:
        # evaluate model performance
        eval_performance(model, X_test, y_test, 
                         join(args.out, f"{args.ix}_performance.csv"), 
                         args.celltype, 
                         logvar=args.logvar)

        # save model and history
        # model.save(join(args.out, f"{args.ix}_lentiMPRA.h5"))
        model.save_weights(f'{args.out}/{args.ix}_ResidualBind.h5') # save weights
        with open(join(args.out, f"{args.ix}_historyDict"), 'wb') as pickle_fh:
            pickle.dump(history.history, pickle_fh)

    # save updated config as yaml
    with open(join(args.out, "config.yaml"), 'w') as f:
        yaml.dump(dict(wandb.config), f, allow_unicode=True, default_flow_style=False)

    wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)