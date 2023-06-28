"""

   Copyright 2023 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
# If EXACT reproducible results are needed
# rseed = 2021
# from numpy.random import seed
# seed(rseed)
# from tensorflow import set_random_seed
# set_random_seed(rseed)

# Generic python
import argparse
import pdb
import os
import sys
import json
import pickle
import time

# Data science ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model

# Custom packages
from detector import lstm, cnn, gru
from data_loader import load_train_data, load_test_data

import utils

def train_forecast_model(model_type, config, Xtrain, Xval, Ytrain, Yval):

    train_params = config['train']
    model_params = config['model']

    # define model input size parameter --- needed for AE size
    model_params['nI'] = Xtrain.shape[2]

    if model_type == 'GRU':
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == 'LSTM':
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == 'CNN':
        event_detector = cnn.ConvNN(**model_params)
    else:
        print(f'Model type {model_type} is not supported.')
        return

    event_detector.create_model()
    
    event_detector.train(Xtrain, Ytrain,
            validation_data=(Xval, Yval),
            **train_params)

    return event_detector

def train_forecast_model_by_idxs(model_type, config, Xfull, train_idxs, val_idxs):

    train_params = config['train']
    model_params = config['model']

    # define model input size parameter --- needed for AE size
    model_params['nI'] = Xfull.shape[1]

    if model_type == 'GRU':
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == 'LSTM':
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == 'CNN':
        event_detector = cnn.ConvNN(**model_params)
    else:
        print(f'Model type {model_type} is not supported.')
        return

    event_detector.create_model()
    
    event_detector.train_by_idx(Xfull, train_idxs, val_idxs,
            validation_data=True,
            **train_params)

    return event_detector

def save_model(event_detector, config, run_name='results'):
    model_name = config['name']
    try:
        event_detector.save(f'models/{run_name}/{model_name}')
    except FileNotFoundError:
        event_detector.save(f'models/results/{model_name}')
        print(f"Directory models/{run_name}/ not found, model {model_name} saved at models/results/ instead")
        print(f"Note: we recommend creating models/{run_name}/ to store this model")

# functions
def load_saved_model(model_type, params_filename, model_filename):
    """ Load stored model. """

    # load params and create event detector
    with open(params_filename) as fd:
        model_params = json.load(fd)

    if model_type == 'CNN':
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == 'LSTM':
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == 'GRU':
        event_detector = gru.GatedRecurrentUnit(**model_params)
    else:
        print(f'Model type {model_type} is not supported.')
        return None

    # load keras model
    event_detector.inner = load_model(model_filename)

    return event_detector

def parse_arguments():

    parser = utils.get_argparser()

    ### Train Params
    parser.add_argument("--train_params_epochs",
        default=100,
        type=int,
        help="Number of training epochs")
    parser.add_argument("--train_params_batch_size",
        default=512,
        type=int,
        help="Training batch size. Note: MUST be larger than history/window values given")
    parser.add_argument("--train_params_no_callbacks",
        action='store_true',
        help="Remove callbacks like early stopping")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

    train_params = {
        'batch_size': args.train_params_batch_size,
        'epochs': args.train_params_epochs,
        'use_callbacks': not args.train_params_no_callbacks,
        'steps_per_epoch': 0,
        'validation_steps': 0,
        'verbose' : 1
    }

    config = {
        # 'grid_search': {
        #     'percentile': args.detect_params_percentile,
        #     'window': args.detect_params_windows,
        #     'metrics': args.detect_params_metrics,
        #     'pr-plot': False,
        #     'detection-plots': args.detect_params_plots,
        #     'save-metric-info': args.detect_params_save_npy
        # }
    }

    run_name = args.run_name
    # test_split = args.detect_params_test_split
    utils.update_config_model(args, config, model_type, dataset_name)
    model_name = config['name']

    Xfull, sensor_cols = load_train_data(dataset_name)
    Xtest, Ytest, _ = load_test_data(dataset_name)

    history = config['model']['history']

    train_idxs, val_idxs = utils.train_val_history_idx_split(dataset_name, Xfull, history)

    train_params['steps_per_epoch'] = len(train_idxs) // train_params['batch_size']
    train_params['validation_steps'] = len(val_idxs) // train_params['batch_size']
    config.update({'train': train_params})
    
    event_detector = train_forecast_model_by_idxs(model_type, config, Xfull, train_idxs, val_idxs)

    save_model(event_detector, config, run_name=run_name)

    print("Finished!")
