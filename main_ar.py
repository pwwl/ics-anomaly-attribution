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

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

# Custom packages
from data_loader import load_train_data, load_test_data

import attack_utils
import utils
import tep_utils

def parse_arguments():

    parser = utils.get_argparser()
    return parser.parse_args()

def train_by_idx(model, Xfull, history, train_idxs, val_idxs):
   
    batch_size = 10000
    
    # Generic data generator object for feeding data to fit_generator
    def data_generator(X, idxs, bs):
        
        i = 0
        while True:
            i += bs

            # Restart from beginning
            if i + bs > len(idxs):
                i = 0 

            X_batch = []
            Y_batch = []

            # Build the history out by sampling from the list of idxs
            for b in range(bs):
                lead_idx = idxs[i+b]
                X_batch.append(X[lead_idx-history:lead_idx])
                Y_batch.append(X[lead_idx+1])

            yield (np.array(X_batch), np.array(Y_batch))

    train_params = dict()
    
    train_params['epochs'] = 20
    train_params['steps_per_epoch'] = len(train_idxs) // batch_size
    train_params['validation_steps'] = len(val_idxs) // batch_size
    train_params['callbacks'] = EarlyStopping(monitor='val_loss', patience=3, verbose=0,  min_delta=0, mode='auto', restore_best_weights=True)
    train_params['validation_data'] = data_generator(Xfull, val_idxs, batch_size)
    
    model.fit(data_generator(Xfull, train_idxs, batch_size), **train_params)

def train_ar(dataset_name):

    Xfull, sensor_cols = load_train_data(dataset_name)
    history = 10

    all_idxs = np.arange(history, len(Xfull)-1)
    train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
    models = []

    for col_idx in range(len(sensor_cols)):

        print(f'Training col {col_idx}')
        Xcol = Xfull[:,col_idx]
        input_layer = Input(shape=(history))
        output_layer = Dense(1, activation=None)(input_layer)
        
        # Define the total model
        model = Model(input_layer, output_layer)
        model.compile(loss='mean_squared_error', optimizer='adam')
        train_by_idx(model, Xcol, history, train_idxs, val_idxs)
        models.append(model)

    pickle.dump(models, open(f'model-AR-{dataset_name}.pkl', 'wb'))

def test_ar(Xtest, dataset_name, footer):

    models = pickle.load(open(f'model-AR-{dataset_name}.pkl', 'rb'))

    ar_scores = np.zeros((len(Xtest) - 10 - 1, Xtest.shape[1]))
    Xwindow, Ywindow = utils.transform_to_window_data(Xtest, Xtest, 10) 

    for j in range(Xtest.shape[1]):
        
        print(f'AR on col {j}')
        model_ar = models[j]

        bs = 20000
        for i in range(0, len(Xwindow), bs):

            X_src = Xwindow[i:i+bs,:,j]
            Y_src = Ywindow[i:i+bs,j]

            Xpred = model_ar.predict(X_src).reshape(-1)            
            err = (Xpred - Y_src)**2
            ar_scores[i:i+bs, j] = err

    np.savetxt(f'AR-scores-{dataset_name}-{footer}.csv', ar_scores, delimiter=',')

def test_val_ar(dataset_name):

    Xfull, sensor_cols = load_train_data(dataset_name)
    history = 10

    all_idxs = np.arange(history, len(Xfull)-1)
    train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
    models = pickle.load(open(f'model-AR-{dataset_name}.pkl', 'rb'))

    Xval = np.zeros((len(val_idxs), history, len(sensor_cols)))
    Yval = np.zeros((len(val_idxs), len(sensor_cols)))

    for i in range(len(val_idxs)):
        Xval[i] = Xfull[val_idxs[i]-history : val_idxs[i]]
        Yval[i] = Xfull[val_idxs[i]]

    ar_val_scores = np.zeros((len(Xval), len(sensor_cols)))
    
    for j in range(ar_val_scores.shape[1]):
        
        print(f'val AR on col {j}')
        model_ar = models[j]

        bs = 20000
        for i in range(0, len(Xval), bs):

            X_src = Xval[i:i+bs,:,j]
            Y_src = Yval[i:i+bs,j]

            Xpred = model_ar.predict(X_src).reshape(-1)            
            err = (Xpred - Y_src)**2
            ar_val_scores[i:i+bs, j] = err

    np.savetxt(f'AR-val-scores-{dataset_name}.csv', ar_val_scores, delimiter=',')
    pdb.set_trace()

def ar_detection_points(dataset_name):

    print(f'AR for {dataset_name}')
    ar_val = np.loadtxt(f'AR-val-scores-{dataset_name}.csv', delimiter=',')
    history = 10
    cutoff = np.quantile(np.mean(ar_val, axis=1), 0.995)
    detection_lookup = dict()

    if dataset_name == 'TEP':

        footer_list = tep_utils.get_footer_list(patterns=['cons'])
        
        for atk_footer in footer_list:
            
            test_errors = np.loadtxt(f'AR-scores-TEP-{atk_footer}.csv', delimiter=',')
            test_instance_errors = np.mean(test_errors, axis=1)

            # Shift by history to account for initial inputs
            att_start = 10000 - history
            att_end = 14000 - history

            attack_region = test_instance_errors[att_start:att_end]    

            if np.sum(attack_region > cutoff):
                print(f'AR detected atk {atk_footer}')
                detection_lookup[atk_footer] = np.min(np.where(attack_region > cutoff)[0])

    else:

        ar_test = np.loadtxt(f'AR-scores-{dataset_name}.csv', delimiter=',')
        test_instance_errors = np.mean(ar_test, axis=1)

        attacks, _ = attack_utils.get_attack_indices(dataset_name)

        for atk_idx in range(len(attacks)):
            
            # Shift by history to account for initial inputs
            attack_idxs = attacks[atk_idx]
            att_start = attack_idxs[0] - history
            att_end = attack_idxs[-1] - history

            attack_region = test_instance_errors[att_start:att_end]    

            if np.sum(attack_region > cutoff):
                
                print(f'AR detected atk {atk_idx} length {len(attack_idxs)}')
                detection_lookup[atk_idx] = np.min(np.where(attack_region > cutoff)[0])

    return detection_lookup

def get_ar_detection_points():

    for dataset_name in ['SWAT', 'WADI', 'TEP']:

        lookup = ar_detection_points(dataset_name)
        full_lookup[f'AR-{dataset_name}'] = lookup

    pickle.dump(full_lookup, open('detection-points-AR.pkl' ,'wb'))

if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset
    run_name = args.run_name
    
    full_lookup = dict()

    train_ar(dataset_name)
    test_val_ar(dataset_name)

    if dataset_name == 'TEP':

        footers = tep_utils.get_footer_list(patterns=['cons'])
        for footer in footers:
            Xtest, _, _ = tep_utils.load_tep_attack(dataset_name, footer)
            test_ar(Xtest[:20000], dataset_name, footer)

    else:

        Xtest, _, _ = load_test_data(dataset_name)
        test_ar(Xtest, dataset_name, 'all')

    print("Finished!")
