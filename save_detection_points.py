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
import pickle
import pdb
import sys

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# Data and ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom local packages
import attack_utils
import tep_utils

def get_detection_points(lookup_name, dataset_name):

	history = 50    
	validation_errors = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-{dataset_name}-ns.npy')
	test_errors = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{dataset_name}-ns.npy')
	attacks, labels = attack_utils.get_attack_indices(dataset_name)

	validation_instance_errors = np.mean(validation_errors, axis=1)
	test_instance_errors = np.mean(test_errors, axis=1)
	cutoff = np.quantile(validation_instance_errors, 0.9995)
	
	detection_lookup = dict()
	detection_full_lookup = dict()

	for atk_idx in range(len(attacks)):
		
		# Shift by history to account for initial inputs
		attack_idxs = attacks[atk_idx]
		att_start = attack_idxs[0] - history
		att_end = attack_idxs[-1] - history

		attack_region = test_instance_errors[att_start:att_end]    

		# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
		# ax.plot(attack_region)
		# ax.hlines(cutoff, xmin=0, xmax=len(attack_region), colors='red', linestyles='dashed')
		# ax.set_title(f'{dataset_name} attack {atk_idx}')
		# plt.savefig(f'{lookup_name}-attack{atk_idx}.png')
		# plt.close()

		if np.sum(attack_region > cutoff) > 0:
			det_point = np.min(np.where(attack_region > cutoff)[0])
			detection_lookup[atk_idx] = det_point
			detection_full_lookup[atk_idx] = np.where(attack_region > cutoff)[0]
			print(f'{lookup_name} detected atk {atk_idx} length {len(attack_idxs)} at point {det_point}')
		else:
			print(f'{lookup_name} missed atk {atk_idx}')

		#pdb.set_trace()
	
	return detection_lookup, detection_full_lookup

def get_tep_detection_points(lookup_name, dataset_name):

	history = 50    
	validation_errors = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-{dataset_name}-ns.npy')
	validation_instance_errors = np.mean(validation_errors, axis=1)

	footer_list1 = tep_utils.get_footer_list(patterns=['cons'])
	footer_list2 = tep_utils.get_footer_list(patterns=['csum', 'line'], mags=['p2s'])
	
	footer_list = footer_list1 + footer_list2
	cutoff = np.quantile(validation_instance_errors, 0.9995)
	
	detection_lookup = dict()
	detection_full_lookup = dict()

	for atk_footer in footer_list:
		
		test_errors = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{atk_footer}-ns.npy')
		test_instance_errors = np.mean(test_errors, axis=1)

		# Shift by history to account for initial inputs
		att_start = 10000 - history
		att_end = 14000 - history

		attack_region = test_instance_errors[att_start:att_end]    

		if np.sum(attack_region > cutoff) > 0:
			print(f'{lookup_name} detected atk {atk_footer}')
			detection_lookup[atk_footer] = np.min(np.where(attack_region > cutoff)[0])
			detection_full_lookup[atk_footer] = np.where(attack_region > cutoff)[0]
	
	return detection_lookup, detection_full_lookup


if __name__ == "__main__":

	datasets = ['SWAT', 'WADI', 'TEP']
	models = ['CNN', 'GRU', 'LSTM']
	run_name = 'results_ns1'
	
	model_detection_lookup = dict()
	model_detection_full_lookup = dict()

	for dataset_name in datasets:
		for model_type in models:
		
			if model_type == 'CNN':
				lookup_name = f'CNN-{dataset_name}-l2-hist50-kern3-units64-{run_name}'
			else:
				lookup_name = f'{model_type}-{dataset_name}-l2-hist50-units64-{run_name}'

			if dataset_name == 'TEP':
				detection_lookup, detection_full_lookup = get_tep_detection_points(lookup_name, dataset_name)
			else:
				detection_lookup, detection_full_lookup = get_detection_points(lookup_name, dataset_name)
			
			model_detection_lookup[lookup_name] = detection_lookup
			model_detection_full_lookup[lookup_name] = detection_full_lookup

	pickle.dump(model_detection_lookup, open('meta-storage/detection-points.pkl' ,'wb'))
	pickle.dump(model_detection_full_lookup, open('meta-storage/all-detection-points.pkl' ,'wb'))

	pdb.set_trace()

	print(f"Finished")
