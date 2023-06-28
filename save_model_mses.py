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
from typing import Dict, List
import argparse
import json
import pickle
import os
import pdb
import sys

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# Data and ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Custom local packages
from data_loader import load_train_data, load_test_data
from main_train import load_saved_model
import metrics
import utils
import tep_utils

def parse_arguments():

    parser = utils.get_argparser()

    parser.add_argument("--detect_params_metrics",
            default=['F1'],
            nargs='+',
            type=str,
            help="Metrics to look over")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Generic evaluation settings
    config = {
        'eval': []
    }

    run_name = args.run_name
    utils.update_config_model(args, config, model_type, dataset_name)

    model_name = config['name']
    lookup_name = f'{model_name}-{run_name}'
    history = config['model']['history']

    Xfull, _ = load_train_data(dataset_name)
    event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')

    if dataset_name == 'TEP':
        
        # Build via indexs
        all_idxs = np.arange(history, len(Xfull)-1)
        _, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)

        ##### Cross Validation
        print('Getting detection errors....')
        validation_errors = utils.reconstruction_errors_by_idxs(Xfull, val_idxs, history)
        print(f'Avg val err: {np.mean(validation_errors)}')

        np.save(f'mses-val-{model_name}-{run_name}-{dataset_name}-ns.npy', validation_errors)
        footers = tep_utils.get_footer_list(patterns=['cons'])
        
        # For TEP, we do each attack separate, since they are in separate files
        for attack_footer in footers:
        
            print(f'scoring {attack_footer} on {model_name} {run_name}')
            Xtest, _, _ = tep_utils.load_tep_attack(dataset_name, attack_footer)
            test_errors = event_detector.reconstruction_errors(Xtest, batches=True, verbose=0)
            np.save(f'mses-{model_name}-{run_name}-{attack_footer}-ns.npy', test_errors)

    else:
    
        Xtest, _, _ = load_test_data(dataset_name)

        # Build via indexs
        all_idxs = np.arange(history, len(Xfull)-1)
        _, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)

        ##### Cross Validation
        print('Getting detection errors....')
        validation_errors = utils.reconstruction_errors_by_idxs(event_detector, Xfull, val_idxs, history)
        test_errors = event_detector.reconstruction_errors(Xtest, batches=True)

        print(f'Avg val err: {np.mean(validation_errors)}')
        np.save(f'mses-val-{model_name}-{run_name}-ns.npy', validation_errors)
        np.save(f'mses-{model_name}-{run_name}-ns.npy', test_errors)
        print(f'Saved mses-val-{model_name}-{run_name}-ns.npy')
        print(f'Saved mses-{model_name}-{run_name}-ns.npy')

    print("Finished!")
