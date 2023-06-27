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
import attack_utils
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

def create_adv_example(event_detector, dataset_name, cutoff):

    Xtest, _, sensor_cols = load_test_data(dataset_name)
    attacks, labels = attack_utils.get_attack_indices(dataset_name)
    history = 50
    it = 0
    evade_mse_explain = True

    if dataset_name == 'SWAT':

        # Focus on attack #5
        attack_region = Xtest[attacks[5]-history]
        detect_point = 10
        attack_col = sensor_cols.index(labels[5][0])
        orig_focus_region = attack_region[detect_point:detect_point+52]

        # Load from pre-cached example
        # it = 15
        # orig_focus_region = np.load(f'advexample-expl-CNN-SWAT-l2-hist50-kern3-units64-results_ns1-{dataset_name}-it{it}.npy')

    elif dataset_name == 'WADI':

        # Focus on attack #6
        attack_region = Xtest[attacks[6]-history]
        detect_point = 4
        orig_focus_region = attack_region[detect_point:detect_point+52]
        attack_col = sensor_cols.index(labels[6][0])

        # Load from pre-cached example
        #it = 19
        #orig_focus_region = np.load(f'advexample-expl-CNN-WADI-l2-hist50-kern3-units64-results_ns1-{dataset_name}-it{it}.npy')

    elif dataset_name == 'TEP':

        # Focus on attack #6
        Xtest, _, sensor_cols = tep_utils.load_tep_attack(dataset_name, 'cons_p2s_s11')
        detect_point = 297
        att_start = 10000
        attack_region = Xtest[att_start-history:14000]
        orig_focus_region = attack_region[detect_point:detect_point+52]
        attack_col = tep_utils.sen_to_idx('s11')

        # Load from pre-cached example
        #it = 18
        #orig_focus_region = np.load(f'advexample-expl-CNN-TEP-l2-hist50-kern3-units64-results_ns1-{dataset_name}-it{it}.npy')

    # If true, do an evasion on the explanation ranking, rather than detection
    if evade_mse_explain:
        original_errors = event_detector.reconstruction_errors(orig_focus_region)[0]
        original_error = original_errors[attack_col]
        original_rank = tep_utils.scores_to_rank(original_errors, attack_col)
        print(f'Initial error: {original_error} initial rank {original_rank}')
    else:
        original_error = np.mean(event_detector.reconstruction_errors(orig_focus_region))
        print(f'Initial error: {original_error} cutoff {cutoff}')

    best_focus_region = orig_focus_region.copy()
    changelog = list()
    val_range = np.arange(0, 2.2, 0.4) - 1
    best_error = original_error
    evaded = False

    while evaded == False:

        max_error_diff = 0
        best_change = None

        for i in range(51):
            for j in range(len(sensor_cols)):
                for val in val_range:

                    # Don't perturb the attack itself
                    if j == attack_col:
                        continue

                    if dataset_name == 'SWAT' and 'IT' not in sensor_cols[j]:
                        continue

                    if dataset_name == 'WADI' and 'STATUS' in sensor_cols[j]:
                        continue

                    if dataset_name == 'TEP' and 'a' in tep_utils.idx_to_sen(j):
                        continue

                    # Scan over model inputs and find a change that lowers error
                    focus_region = best_focus_region.copy()
                    focus_region[i, j] = val

                    if evade_mse_explain:
                        full_errors = event_detector.reconstruction_errors(focus_region, verbose=0)[0]
                        error = full_errors[attack_col]
                    else:
                        error = np.mean(event_detector.reconstruction_errors(focus_region, verbose=0))

                    error_diff = best_error - error

                    if error_diff > max_error_diff:
                        max_error_diff = error_diff
                        print(f'Iter {it}: change {i},{j},{val} reduces error {max_error_diff}')
                        best_change = (i, j, val)

                    # Prevents OOM
                    del focus_region

        if best_change is not None:

            best_focus_region[best_change[0], best_change[1]] = best_change[2]
            best_errors = event_detector.reconstruction_errors(best_focus_region)[0]
            best_error = best_errors[attack_col]
            new_rank = tep_utils.scores_to_rank(best_errors, attack_col)

            if evade_mse_explain:
                print(f'Iter {it}: best change: {best_change}. New error {best_error} New rank {new_rank}')
            else:
                print(f'Iter {it}: best change: {best_change}. New error {best_error} Cutoff {cutoff}')

            changelog.append(best_change)
            it += 1

            #np.save(f'advexample-{model_name}-{run_name}-{dataset_name}-it{it}.npy', best_focus_region)
            np.save(f'advexample-expl-{model_name}-{run_name}-{dataset_name}-it{it}.npy', best_focus_region)

            if new_rank > 10 or it > 20:
                evaded = True

    pdb.set_trace()

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

    Xfull, sensor_cols = load_train_data(dataset_name)
    event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')

    ##### Cross Validation
    print('Getting detection errors....')
    validation_errors = np.load(f'ccs-storage/mses-val-{model_name}-{run_name}-{dataset_name}-ns.npy')
    cutoff = np.quantile(np.mean(validation_errors, axis=1), 0.9995)

    create_adv_example(event_detector, dataset_name, cutoff)

    print("Finished!")
