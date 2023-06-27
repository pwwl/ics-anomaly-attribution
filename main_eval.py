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

SHADE_OF_GRAY = '0.1'

def eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics=['F1'], percentile=0.995, window=1, plot=False, best=False):

    if best:
        Yhat = event_detector.best_cached_detect(test_errors)
        used_window = event_detector.params['best_window']
        used_theta = event_detector.params['best_theta']
        Yhat = Yhat[used_window-1:].astype(int)
    else:
        theta = np.quantile(val_errors, percentile)
        Yhat = event_detector.cached_detect(test_errors, theta = theta, window = window)
        Yhat = Yhat[window-1:].astype(int)
        used_window = window
        used_theta = theta

    # Final test performance
    for metric in eval_metrics:

        metric_func = metrics.get(metric)
        Yhat_copy, Ytest_copy = utils.normalize_array_length(Yhat, Ytest)
        final_value = metric_func(Yhat_copy, Ytest_copy)
        print(f'At theta={used_theta}, window={used_window}, {metric}={final_value}')

    if plot:

        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(-1 * Yhat_copy, color = '0.25', label = 'Predicted')
        ax.plot(Ytest_copy, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
        ax.fill_between(np.arange(len(Yhat_copy)), -1 * Yhat_copy, 0, color = '0.25')
        ax.fill_between(np.arange(len(Ytest_copy)), 0, Ytest_copy, color = 'lightcoral')
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels(['Predicted','Benign','Attacked'])
        fig.savefig(f'eval-detection.pdf')

    return Yhat_copy, Ytest_copy

# Compare two settings side by side
def eval_demo(event_detector, model_type, config, val_errors, test_errors, Ytest, eval_metrics=['F1'], run_name='results', include_best = True):

    model_name = config['name']
    eval_config = config['eval']

    if include_best:
        eval_config.append('best')

    # Can't use indexed subplots when length is 1
    if len(eval_config) < 2:

        fig, ax = plt.subplots(figsize=(20, 4))

        if eval_config[0] == 'best':
            Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, best=True)
            ax.set_title(f'Detection trajectory on test dataset with best parameters', fontsize = 36)
        else:
            percentile = eval_config[0]['percentile']
            window = eval_config[0]['window']
            ax.set_title(f'Detection trajectory on test dataset, percentile={percentile}, window={window}', fontsize = 36)
            Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, percentile=percentile, window=window)

        ax.plot(-1 * Yhat, color = '0.25', label = 'Predicted')
        ax.plot(Ytest, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
        ax.fill_between(np.arange(len(Yhat)), -1 * Yhat, 0, color = '0.25')
        ax.fill_between(np.arange(len(Ytest)), 0, Ytest, color = 'lightcoral')
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels(['Predicted','Benign','Attacked'])

    else:

        fig, ax = plt.subplots(len(eval_config), figsize=(20, 4 * len(eval_config)))
        all_Yhats = []

        for i in range(len(eval_config)):

            if eval_config[i] == 'best':
                Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, best=True)
                ax[i].set_title(f'Detection trajectory on test dataset, with best parameters', fontsize = 36)
            else:
                percentile = eval_config[i]['percentile']
                window = eval_config[i]['window']
                ax[i].set_title(f'Detection trajectory on test dataset, percentile={percentile}, window={window}', fontsize = 36)
                Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, percentile=percentile, window=window)

            all_Yhats.append(Yhat)

            ax[i].plot(-1 * Yhat, color = '0.25', label = 'Predicted')
            ax[i].plot(Ytest, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
            ax[i].fill_between(np.arange(len(Yhat)), -1 * Yhat, 0, color = '0.25')
            ax[i].fill_between(np.arange(len(Ytest)), 0, Ytest, color = 'lightcoral')
            ax[i].set_yticks([-1,0,1])
            ax[i].set_yticklabels(['Predicted','Benign','Attacked'])

        pickle.dump(all_Yhats, open(f'{model_name}-eval-Yhats.pkl', 'wb'))
        print('Dumped to pkl.')

    plt.tight_layout(rect=[0, 0, 1, 0.925])
    plt.savefig(f'{model_name}-compare.pdf')


def hyperparameter_eval(event_detector, model_type, config, val_errors, test_errors, Ytest,
    eval_metrics=['F1'],
    cutoffs=['0.995'],
    windows=[1],
    run_name='results'):

    model_name = config['name']
    do_batches = False
    all_Yhats = []

    for metric in eval_metrics:

        # FP is a negative metric (lower is better)
        negative_metric = (metric == 'FP')

        # FP is a negative metric (lower is better)
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_metric = -1000
        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get(metric)

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(val_errors, percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]
                Yhat = event_detector.cached_detect(test_errors, theta = theta, window = window)
                Yhat = Yhat[window-1:].astype(int)

                Yhat_copy, Ytest_copy = utils.normalize_array_length(Yhat, Ytest)
                choice_value = metric_func(Yhat_copy, Ytest_copy)
                metric_vals[percentile_idx, window_idx] = choice_value

                # FPR is a negative metric (lower is better)
                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

        print("Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(metric, best_metric, best_percentile, best_window))

        best_theta = np.quantile(val_errors, best_percentile)
        final_Yhat = event_detector.cached_detect(test_errors, theta = best_theta, window = best_window)
        final_Yhat = final_Yhat[best_window-1:].astype(int)

        metric_func = metrics.get(metric)

        final_Yhat_copy, Ytest_copy = utils.normalize_array_length(final_Yhat, Ytest)
        final_value = metric_func(final_Yhat_copy, Ytest_copy)

        all_Yhats.append(final_Yhat_copy)
        np.save(f'npys/{run_name}/{model_name}-{metric}.npy', metric_vals)
        print(f'Saved {run_name}/{model_name}-{metric}.npy')

        print("Final {} is {:.3f} at percentile={:.5f}, window {}".format(metric, final_value, best_percentile, best_window))
        config['eval'].append({'percentile': best_percentile, 'window': best_window})

    pickle.dump(all_Yhats, open(f'{model_name}-metrics-Yhats.pkl', 'wb'))
    print('Dumped to pkl.')

    return best_percentile, best_window

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
    } # type: Dict[str, List]

    run_name = args.run_name
    utils.update_config_model(args, config, model_type, dataset_name)

    model_name = config['name']
    Xfull, sensor_cols = load_train_data(dataset_name)
    Xtest, Ytest, _ = load_test_data(dataset_name)
    
    Xfull_window, Yfull = utils.transform_to_window_data(Xfull, Xfull, config['model']['history'])
    _, Xval, _, Yval = train_test_split(Xfull_window, Yfull, test_size=0.2, random_state=42, shuffle=True)

    event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')

    do_batches = False

    if not model_type == 'AE':

        # Clip the prediction to match LSTM prediction window
        Ytest = Ytest[event_detector.params['history'] + 1:]
        do_batches = True

    ##### Cross Validation
    print('Getting detection errors....')
    validation_errors = (event_detector.predict(Xval, verbose=0) - Yval)**2
    test_errors = event_detector.reconstruction_errors(Xtest, batches=do_batches, verbose=0)

    validation_instance_errors = validation_errors.mean(axis=1)
    test_instance_errors = test_errors.mean(axis=1)

    default_cutoffs = [0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.99995]
    default_windows = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    hyper = True
    plot = False

    if hyper:

        bestp, bestw = hyperparameter_eval(event_detector,
            model_type,
            config,
            validation_instance_errors,
            test_instance_errors,
            Ytest,
            eval_metrics=args.detect_params_metrics,
            cutoffs=default_cutoffs,
            windows=default_windows,
            run_name=run_name)

    if plot:

        # Trains and returns the inner event detection model
        eval_demo(event_detector,
            model_type,
            config,
            validation_instance_errors,
            test_instance_errors,
            Ytest,
            eval_metrics=args.detect_params_metrics,
            run_name=run_name,
            include_best=False)

    print("Finished!")
