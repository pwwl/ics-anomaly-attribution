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

import pdb
import pickle
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_train_data(dataset_name, scaler=None, no_transform=False, verbose=False):

    if verbose:
        print('Loading {} train data...'.format(dataset_name))

    if scaler is None:
        if verbose:
            print("No scaler provided. Using default sklearn.preprocessing.StandardScaler")
        scaler = StandardScaler()

    if dataset_name == 'TEP':
        
        df_train = pd.read_csv("data/TEP/TEP_train.csv", dayfirst=True)
        sensor_cols = [col for col in df_train.columns if col not in ['Atk']]
    
    elif dataset_name == 'SWAT':
        
        df_train = pd.read_csv("data/" + dataset_name + "/SWATv0_train.csv", dayfirst=True)
        sensor_cols = [col for col in df_train.columns if col not in ['Timestamp', 'Normal/Attack']]

    elif dataset_name == 'WADI':

        df_train = pd.read_csv("data/" + dataset_name + "/WADI_train.csv")
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW']
        sensor_cols = [col for col in df_train.columns if col not in remove_list]

    else:
        raise ValueError('Dataset name not found.')
        
    # scale sensor data
    if no_transform:
        X = pd.DataFrame(index = df_train.index, columns = sensor_cols, data = df_train[sensor_cols].values)
    else:
        X_prescaled = df_train[sensor_cols].values
        X = pd.DataFrame(index = df_train.index, columns = sensor_cols, data = scaler.fit_transform(X_prescaled))

        # Need fitted scaler for future attack/test data        
        pickle.dump(scaler, open(f'models/{dataset_name}_scaler.pkl', 'wb'))
        if verbose:
            print('Saved scaler parameters to {}.'.format('scaler.pkl'))

    return X.values, sensor_cols

def load_test_data(dataset_name, scaler=None, no_transform=False, verbose=False):

    if verbose:
        print('Loading {} test data...'.format(dataset_name))

    if scaler is None:
        if verbose:
            print('No scaler provided, trying to load from models directory...')
        scaler = pickle.load(open(f'models/{dataset_name}_scaler.pkl', "rb"))
        print('Successful.')

    if dataset_name == 'TEP':
        
        df_test = pd.read_csv("data/" + dataset_name + "/TEP_test.csv", dayfirst=True)
        sensor_cols = [col for col in df_test.columns if col not in ['Atk']]
        target_col = 'Atk'

    elif dataset_name == 'SWAT':
        df_test = pd.read_csv("data/" + dataset_name + "/SWATv0_test.csv")
        sensor_cols = [col.strip() for col in df_test.columns if col not in ['Timestamp', 'Normal/Attack']]
        target_col = 'Normal/Attack'

    elif dataset_name == 'WADI':
        
        df_test = pd.read_csv("data/" + dataset_name + "/WADI_test.csv")
        
        # Remove nan columns
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW']
        sensor_cols = [col for col in df_test.columns if col not in remove_list]
        target_col = 'Attack'

    else:
        raise ValueError('Dataset name not found.')
        return

    # scale sensor data
    if no_transform:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
    else:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
    
    Ytest = df_test[target_col]

    return Xtest.values, Ytest.values, sensor_cols
