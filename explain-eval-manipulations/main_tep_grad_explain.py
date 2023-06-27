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

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import pickle
import pdb
import sys

sys.path.append('..')

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_loader import load_train_data, load_test_data
from grad_explainer import smooth_grad_mse_explainer, integrated_gradients_mse_explainer, expected_gradients_mse_explainer
from grad_explainer import smooth_grad_explainer, integrated_gradients_explainer
from main_train import load_saved_model

from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list

import metrics
import utils

att_skip_list = get_skip_list()

def explain_true_position(event_detector, run_name, model_name, explainer, Xtest, baseline, footer, use_top_feat=False, num_samples=100):

	attack_start = 10000
	attack_end = 14000

	history = event_detector.params['history']
	gif_outputs = np.zeros((num_samples, history, Xtest.shape[1]))

	count = 0

	for i in range(num_samples):

		# Go at least 1 history deep in the history
		att_start = attack_start + i - history - 1
		att_end = attack_start + i + 1

		Xattack = Xtest[att_start : att_end]
		Xattack_src, Yattack_src = event_detector.transform_to_window_data(Xattack, Xattack)

		print(f'For attack {footer}, Processing {att_start} to {att_end}')

		if use_top_feat:

			rec_errors = event_detector.reconstruction_errors(Xattack, batches=False)
			top_feat = np.argmax(rec_errors)

			# Only the first true positive is taken
			explainer.setup_explainer(event_detector.inner, Yattack_src, top_feat)
			gif_output = explainer.explain(Xattack_src, baselines=baseline, multiply_by_input=True)
			gif_outputs[i] = gif_output

		else:

			# Only the first true positive is taken
			explainer.setup_explainer(event_detector.inner, Yattack_src)
			gif_output = explainer.explain(Xattack_src, baselines=baseline, multiply_by_input=True)
			gif_outputs[i] = gif_output

		count += 1

		if count >= num_samples:
			break

	pickle.dump(gif_outputs, open(f'explanations-{explainer.get_name()}-{model_name}-{run_name}-{footer}-true{num_samples}.pkl', 'wb'))

	return

def explain_detect(event_detector, run_name, model_name, explainer, Xtest, baseline, attack_footer, detection_points, use_top_feat=False, num_samples=100):

	attack_start = 10000
	attack_end = 14000

	if attack_footer in detection_points:

		detect_idx = detection_points[attack_footer]
		history = event_detector.params['history']
		gif_outputs = np.zeros((num_samples, history, Xtest.shape[1]))

		count = 0

		for i in range(num_samples):

			capture_idx = attack_start + detect_idx + i
			capture_start = capture_idx - history - 1

			Xattack = Xtest[capture_start : capture_idx+1]
			Xattack_src, Yattack_src = event_detector.transform_to_window_data(Xattack, Xattack)

			print(f'For attack {attack_footer}, Processing {capture_start} to {capture_idx + 1}')

			if use_top_feat:

				rec_errors = event_detector.reconstruction_errors(Xattack, batches=False)
				top_feat = np.argmax(rec_errors)

				# Only the first true positive is taken
				explainer.setup_explainer(event_detector.inner, Yattack_src, top_feat)
				gif_output = explainer.explain(Xattack_src, baselines=baseline, multiply_by_input=True)
				gif_outputs[i] = gif_output

			else:

				# Only the first true positive is taken
				explainer.setup_explainer(event_detector.inner, Yattack_src)
				gif_output = explainer.explain(Xattack_src, baselines=baseline, multiply_by_input=True)
				gif_outputs[i] = gif_output

			count += 1

			if count >= num_samples:
				break

		pickle.dump(gif_outputs, open(f'explanations-{explainer.get_name()}-{model_name}-{run_name}-{attack_footer}-detect{num_samples}.pkl', 'wb'))

	else:
		print(f'Attack {attack_footer} was missed')

	return

def parse_arguments():

	parser = utils.get_argparser()

	parser.add_argument("attack",
		help="Which attack to explore?",
		type=str)

	# Explain specific
	parser.add_argument("--explain_params_methods",
		default=['SM', 'SG'],
		nargs='+',
		type=str,
		help="Which explanation methods to use? Options: [SM, SG, IG, EG]")

	parser.add_argument("--explain_params_use_top_feat",
		action='store_true',
		help="Explain based off top MSE feature, rather than entire MSE")

	parser.add_argument("--explain_params_threshold",
		default=0,
		type=float,
		help="Percentile threshold for selecting candidates for explanation. 0 (default) chooses optimal.")

	return parser.parse_args()

if __name__ == "__main__":

	args = parse_arguments()
	model_type = args.model
	dataset_name = args.dataset
	attack_footer = args.attack

	os.chdir('..')

	if attack_footer in att_skip_list:
		print(f'{attack_footer} is in skip list. returning.....')
		exit(0)

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

	use_top_feat = args.explain_params_use_top_feat
	print(f'Explaining top feature only?: {use_top_feat}')

	run_name = args.run_name
	config = {} # type: Dict[str, str]
	utils.update_config_model(args, config, model_type, dataset_name)

	model_name = config['name']
	history = config['model']['history']

	Xfull, sensor_cols = load_train_data(dataset_name)
	event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')

	use_top_feat = False

	explainer_codes = args.explain_params_methods
	explainers = []

	if use_top_feat:
		if 'SM' in explainer_codes:
			explainers.append(('SM', smooth_grad_explainer.SaliencyMapHistoryExplainer()))
		if 'SG' in explainer_codes:
			explainers.append(('SG', smooth_grad_explainer.SmoothGradHistoryExplainer()))
		if 'IG' in explainer_codes:
			explainers.append(('IG', integrated_gradients_explainer.IntegratedGradientsHistoryExplainer()))
	else:
		if 'SM' in explainer_codes:
			explainers.append(('SM', smooth_grad_mse_explainer.SaliencyMapMseHistoryExplainer()))
		if 'SG' in explainer_codes:
			explainers.append(('SG', smooth_grad_mse_explainer.SmoothGradMseHistoryExplainer()))
		if 'IG' in explainer_codes:
			explainers.append(('IG', integrated_gradients_mse_explainer.IntegratedGradientsMseHistoryExplainer()))
		if 'EG' in explainer_codes:
			# Note: If using expected gradients, the baseline needs to be changed
			explainers.append(('EG', expected_gradients_mse_explainer.ExpectedGradientsMseHistoryExplainer()))

	if 'IG' in explainer_codes or 'EG' in explainer_codes:

		#################################
		# BASELINE FOR INTEGRATED GRADIENTS
		#################################

		# Build via indexs
		all_idxs = np.arange(history, len(Xfull)-1)
		train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
		
		Xbatch = []

		# Build the history out by sampling from the list of idxs
		for b in range(5000):
			lead_idx = val_idxs[b]
			Xbatch.append(Xfull[lead_idx-history:lead_idx])

		Xbatch = np.array(Xbatch)
		avg_benign = np.mean(Xbatch, axis=0)
		baseline = np.expand_dims(avg_benign, axis=0)

		#################################
		# BASELINE FOR EXPECTED GRAD
		#################################

		# If using expected gradients, baseline needs to be training examples. We subsample training regions
		train_samples = []
		num_to_sample = 500
		inc = len(Xbatch) // num_to_sample

		for i in range(num_to_sample):
			train_sample = Xbatch[i * inc]
			train_samples.append(train_sample)

		eg_baseline = np.array(train_samples)

		print('Built baseline objs')

	else:

		baseline = None
		eg_baseline = None

	lookup_name = f'{model_name}-{run_name}'
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
	model_detection_points = detection_points[lookup_name]
	Xtest, Ytest, sensor_cols = load_tep_attack(dataset_name, attack_footer)
	num_samples = 150

	# Each explanation method in outer loop
	for code, expl in explainers:
		print('======================')
		print(f'STARTING EXPLANATIONS WITH {expl.get_name()}')
		print('======================')

		if code == 'EG':
			explain_true_position(event_detector, run_name, model_name, expl, Xtest, eg_baseline, attack_footer, num_samples=5)
			explain_detect(event_detector, run_name, model_name, expl, Xtest, eg_baseline, attack_footer, model_detection_points, num_samples=num_samples)
		else:
			explain_true_position(event_detector, run_name, model_name, expl, Xtest, baseline, attack_footer, num_samples=5)
			explain_detect(event_detector, run_name, model_name, expl, Xtest, baseline, attack_footer, model_detection_points, num_samples=num_samples)

	print("Finished!")
