"""

   Copyright 2020 Lujo Bauer, Clement Fung

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

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
np.set_printoptions(suppress=True)

import argparse
import json
import pickle
import os

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

sys.path.append('..')
from data_loader import load_train_data, load_test_data
import metrics
import utils

HOUR = 2000

def attack_footer_to_sensor_idx(attack_footer):

    splits = attack_footer.split("_")
    sensor_type = splits[2][0]
    sensor_value = int(splits[2][1:])

    if sensor_type == 'a':
        return sensor_value + 40
    elif sensor_type == 's':
        return sensor_value - 1
    else:
        print(f'Something wrong! Found sensor_type {sensor_type}')
        exit()

    return -1

def plot_attack_comparison(dataset_name, sensor_cols, attacks, Xtrain):

    scaler = pickle.load(open(f'models/{dataset_name}_scaler.pkl', "rb"))
    history = 50

    att_start = 5*HOUR - history
    att_end = 7*HOUR

    # Show extra half hour 
    settle_time = int(0.5*HOUR)
    plot_start = att_start - history
    plot_end = att_end + settle_time 

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    titles = ['Summing', 'Linear']

    for at_idx in range(len(attacks)):
        
        attack_footer = attacks[at_idx]
        Xtest, _, _ = load_tep_attack(dataset_name, attack_footer)
        sensor_idx = attack_footer_to_sensor_idx(attack_footer)

        # TODO: A bit clumsy here, since if we use predict() or reconstruction_errors() the history is shifted automatically.
        Xfull = Xtest[plot_start:plot_end]
        
        Xplot = scaler.inverse_transform(Xfull)[:, sensor_idx]
        Xplot_benign = scaler.inverse_transform(Xtrain[plot_start:plot_end])[:, sensor_idx]

        print(f'Plotting source from {plot_start} to {plot_end}')
        print(f'Plotting {sensor_idx} (true attack): {sensor_cols[sensor_idx]}')

        ax[at_idx].set_title(f'{titles[at_idx]} manipulation on {sensor_cols[sensor_idx]}', fontsize=24)
        ax[at_idx].plot(Xplot_benign, color='grey', label='Original', lw=2)
        ax[at_idx].plot(Xplot, color='red', label='Manipulated', lw=2)
        
        #ax[at_idx].vlines(x=[history-1, len(Xplot) - settle_time], ymin=np.min(Xplot), ymax=np.max(Xplot), color='red', ls='--')

        ax[at_idx].grid(True, which='major', axis='y', linestyle = '--')
        ax[at_idx].set_xticks([])
        ax[at_idx].set_ylim([46.5, 49.5])
        ax[at_idx].tick_params(axis='y', labelsize=18)

    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=24)

    fig.tight_layout()
    plt.savefig(f'plot-attack-patterns.pdf')
    plt.close()

    print('Done plot helper')
    return

def load_tep_attack(dataset_name, attack_footer, scaler=None, no_transform=False):

    print('Loading {} test data...'.format(dataset_name))

    if scaler is None:
        print('No scaler provided, trying to load from models directory...')
        scaler = pickle.load(open(f'models/{dataset_name}_scaler.pkl', "rb"))
        print('Successful.')

    if dataset_name == 'TEP':
        df_test = pd.read_csv(f"tep-attacks/matlab/TEP_test_{attack_footer}.csv", dayfirst=True)
        sensor_cols = [col for col in df_test.columns if col not in ['Atk']]
        target_col = 'Atk'

    else:
        print('This script is meant for TEP only.')
        return

    # scale sensor data
    if no_transform:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
    else:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
    
    Ytest = df_test[target_col]

    return Xtest.values, Ytest.values, Xtest.columns

if __name__ == "__main__":
    
    os.chdir('..')

    Xtrain, sensor_cols = load_train_data('TEP')

    an = 's6'
    at = 'p5s'

    attack_patterns = ['csum', 'line']
    attack_footers = []

    for ap in attack_patterns:
        attack_footers.append(f'{ap}_{at}_{an}')
        
    plot_attack_comparison('TEP', sensor_cols, attack_footers, Xtrain)
    
    print("Finished!")
