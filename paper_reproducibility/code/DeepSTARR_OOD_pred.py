import argparse
import utils
from model_zoo import DeepSTARR
from dynamic_aug import DynamicAugModel
import keras
from keras.models import load_model
from os.path import join
import numpy as np
import h5py
import gc
import yaml 

'''
For OOD sequences (provided in h5 file), use DeepSTARR models to make predictions 
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help='path to model')
    parser.add_argument("--out", type=str,
                        help="output directory to save results")
    parser.add_argument("--seqs", type=str,
                        help='h5 file containing ood seqs to predict on')
    parser.add_argument('--load_from_weights', action='store_true',
                        help='if true, model weights are provided')
    parser.add_argument("--config", default=None,
                        help='needed to load model from weights')
    parser.add_argument('--dynamic', action='store_true',
                        help='if true, load DynamicAugModel')
    parser.add_argument('--aug', default=None, action='store',
                        help='one of evoaug/random/mutagenesis')
    args = parser.parse_args()
    return args

def parse_seqs(h5_file):
    '''
    get OOD seqs from h5 file
    return dictionary
    '''

    data = h5py.File(h5_file, 'r')
    X_test_random = np.array(data['random'])
    X_test_mutagenesis = np.array(data['mutagenesis'])
    X_test_evoaug = np.array(data['evoaug'])
    data.close()

    # parse as dict
    ood_seqs_dict = {'random': X_test_random,
                    'mutagenesis': X_test_mutagenesis,
                    'evoaug': X_test_evoaug}
    
    return ood_seqs_dict

def load_model_from_weights(weights, configfile, epistemic=True, dynamic=False, aug=None):
    config = yaml.safe_load(open(configfile, 'r'))
    if dynamic:
        model = DynamicAugModel(DeepSTARR, input_shape=(249,4), aug=aug, config=config, epistemic=True)
    else:
        model = DeepSTARR(input_shape=(249,4), config=config, epistemic=epistemic)


def main(args):
    ood_seqs_dict = parse_seqs(args.seqs)

    if args.load_from_weights:
        model = load_model_from_weights(args.model, args.config, dynamic=args.dynamic, aug=args.aug)
    else:
        model = load_model(args.model)
    
    