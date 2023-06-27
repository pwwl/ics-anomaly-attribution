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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
import shap

import networkx as nx

import os
import pickle
import pdb
import sys

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from sklearn.model_selection import train_test_split

sys.path.append('..')
from data_loader import load_train_data, load_test_data
from live_explainer.score_generator import counterfactual_score_generator, counterfactual_minus_score_generator
from live_explainer.score_generator import mse_score_generator, mse_sd_score_generator
from live_explainer.score_generator import lime_score_generator, shap_score_generator, lemna_score_generator
from grad_explainer import smooth_grad_mse_explainer, integrated_gradients_mse_explainer, expected_gradients_mse_explainer

from main_train import load_saved_model
from tep_utils import get_short_colnames

import attack_utils
import utils

def expl_to_anomaly_score(explain_scores):
	assert (len(explain_scores.shape) == 3)
	return np.sum(np.abs(explain_scores[0]), axis=0)

# TODO convert to general utility
def build_baseline(Xfull, history, method='MSE'):

	# Build via indexs
	all_idxs = np.arange(history, len(Xfull)-1)
	train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
	
	Xbatch = []

	# Build the history out by sampling from the list of idxs
	for b in range(5000):
		lead_idx = val_idxs[b]
		Xbatch.append(Xfull[lead_idx-history:lead_idx])

	Xbatch = np.array(Xbatch)

	if method == 'IG' or method == 'CF-Add' or method == 'CF-Sub':

		#################################
		# BASELINE FOR INTEGRATED GRADIENTS
		#################################

		avg_benign = np.mean(Xbatch, axis=0)
		baseline = np.expand_dims(avg_benign, axis=0)
		print(f'Built {method} baseline obj')

	elif method == 'EG' or method == 'LIME' or method == 'SHAP':

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

		baseline = np.array(train_samples)
		print(f'Built {method} baseline obj')

	else:
		
		baseline = None

	return baseline

def run_adv_ml_unit_test(event_detector, lookup_name, attack_col, expl=None, baselines=None):

	#adv_example = np.load(f'ccs-storage/advexample-{lookup_name}.npy')
	benign_example = np.load(f'ccs-storage/advexample-{lookup_name}-orig.npy')
	adv_example = np.load(f'ccs-storage/advexample-{lookup_name}.npy')
	adv_example_expl = np.load(f'ccs-storage/advexample-{lookup_name}-expl.npy')

	for example in [benign_example, adv_example, adv_example_expl]:

		Xwindow_src = np.expand_dims(example[:50], axis=0)
		Ywindow_src = np.expand_dims(example[-1], axis=0)

		mses = ((event_detector.predict(Xwindow_src) - Ywindow_src)**2)[0]
		
		if method == 'SHAP':
			exp_attribution = shap_score_generator(event_detector, expl, Xwindow_src, Ywindow_src)
			expl_scores = expl_to_anomaly_score(np.expand_dims(exp_attribution, axis=0))    
		elif method == 'LEMNA':
			exp_attribution = lemna_score_generator(event_detector, Xwindow_src, Ywindow_src)
			expl_scores = expl_to_anomaly_score(np.expand_dims(exp_attribution, axis=0))
		else:
			# Generic case for whitebox
			output_mod = expl.explain(Xwindow_src, baselines=baselines, multiply_by_input=True)
			expl_scores = expl_to_anomaly_score(output_mod)

		mses /= np.sum(mses)
		expl_scores /= np.sum(expl_scores)

		mserank = np.where(np.argsort(mses)[::-1] == attack_col)[0] + 1 
		exrank = np.where(np.argsort(expl_scores)[::-1] == attack_col)[0] + 1

		print(f'Adv ML ranks -- MSE: score {mses[attack_col]:.3f} rank {mserank} exp score {expl_scores[attack_col]:.3f} rank {exrank}')

	return 

def run_ar_unit_test(dataset_name, Xwindow_src, Ywindow_src, mag=2, verbose=False):

	mse_ranks = np.zeros(len(sensor_cols))
	mse_props = np.zeros(len(sensor_cols))
	exp_ranks = np.zeros(len(sensor_cols))
	exp_props = np.zeros(len(sensor_cols))
	ar_model = pickle.load(open(f'model-AR-{dataset_name}.pkl', 'rb'))

	for i in range(0, len(sensor_cols)):

		print(f'Test {i}')
		Xwindow_mod = Xwindow_src.copy()
		Xwindow_mod[:, :, i] += (mag + 1e-3)

		ar_score = np.zeros(len(sensor_cols))
		for j in range((len(sensor_cols))):
			ar_score[j] = (ar_model[j].predict(Xwindow_mod[:,-10:,0]) - Ywindow_src[:,j])**2

		expl_choice = np.argmax(ar_score)
		exrank = np.where(np.argsort(ar_score)[::-1] == i)[0] + 1
		exprop = 100 * ar_score[i] / np.sum(ar_score)

		exp_ranks[i] = exrank
		exp_props[i] = exprop

		if verbose:
			if exrank == 1:
				print(f'col {i}: {sensor_cols[i]} passed. EXPrank {exrank} {exprop:.2f}%')
			else:
				print(f'col {i}: {sensor_cols[i]} failed. EXPMank {exrank} {exprop:.2f}% ({sensor_cols[expl_choice]})')

	return exp_ranks, exp_props

def run_unit_test(event_detector, expl, Xwindow_src, Ywindow_src, sensor_cols, dataset, mag=2, baselines=None, verbose=False, method='MSE'):

	mse_ranks = np.zeros(len(sensor_cols))
	mse_props = np.zeros(len(sensor_cols))

	exp_ranks = np.zeros(len(sensor_cols))
	exp_props = np.zeros(len(sensor_cols))

	for i in range(0, len(sensor_cols)):

		Xwindow_mod = Xwindow_src.copy()
		Xwindow_mod[:, :, i] += (mag + 1e-3)
		
		mses = ((event_detector.predict(Xwindow_mod) - Ywindow_src)**2)[0]
		
		if method == 'LIME':
			exp_attribution = lime_score_generator(event_detector, expl, Xwindow_mod, Ywindow_src)
			expl_scores = expl_to_anomaly_score(np.expand_dims(exp_attribution, axis=0))    
		elif method == 'SHAP':
			exp_attribution = shap_score_generator(event_detector, expl, Xwindow_mod, Ywindow_src)
			expl_scores = expl_to_anomaly_score(np.expand_dims(exp_attribution, axis=0))    
		elif method == 'LEMNA':
			exp_attribution = lemna_score_generator(event_detector, Xwindow_mod, Ywindow_src)
			expl_scores = expl_to_anomaly_score(np.expand_dims(exp_attribution, axis=0))    
		elif method == 'CF-Add':
			exp_attribution = counterfactual_score_generator(event_detector, Xwindow_mod, Ywindow_src, baseline=baselines)
			expl_scores = np.abs(exp_attribution)
		elif method == 'CF-Sub':
			exp_attribution = counterfactual_minus_score_generator(event_detector, Xwindow_mod, Ywindow_src, baseline=baselines)
			expl_scores = np.abs(exp_attribution)
		else:
			# Generic case for whitebox
			output_mod = expl.explain(Xwindow_mod, baselines=baselines, multiply_by_input=True)
			expl_scores = expl_to_anomaly_score(output_mod)
			
		mse_choice = np.argmax(mses)
		expl_choice = np.argmax(expl_scores)

		mserank = np.where(np.argsort(mses)[::-1] == i)[0] + 1 
		exrank = np.where(np.argsort(expl_scores)[::-1] == i)[0] + 1

		mseprop = 100 * mses[i] / np.sum(mses)
		exprop = 100 * expl_scores[i] / np.sum(expl_scores)

		mse_props[i] = mseprop
		mse_ranks[i] = mserank

		exp_props[i] = exprop
		exp_ranks[i] = exrank

		if verbose:
			if i == expl_choice:
				print(f'col {i}: {sensor_cols[i]} passed. MSErank {mserank} {mseprop:.2f}% EXPrank {exrank} {exprop:.2f}%')
			else:
				print(f'col {i}: {sensor_cols[i]} failed. MSErank {mserank} {mseprop:.2f}% ({sensor_cols[mse_choice]}) EXPMank {exrank} {exprop:.2f}% ({sensor_cols[expl_choice]})')
		
	return mse_ranks, mse_props, exp_ranks, exp_props

def parse_arguments():

	parser = utils.get_argparser()
	
	# Explain specific
	parser.add_argument("--explain_params_method",
		default=['SM'],
		nargs='+',
		type=str,
		help="Which explanation methods to use? Options: [SM, SG, IG, EG]")
	
	return parser.parse_args()

if __name__ == "__main__":

	import os

	args = parse_arguments()
	model_type = args.model
	dataset_name = args.dataset
	exp_methods = args.explain_params_method

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

	run_name = args.run_name
	config = {}
	utils.update_config_model(args, config, model_type, dataset_name)

	model_name = config['name']
	Xfull, sensor_cols = load_train_data(dataset_name)
	#Xtest, _, sensor_cols = load_test_data(dataset_name)
	event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')
	history = event_detector.params['history']

	if dataset_name == 'TEP':
		sensor_cols = get_short_colnames()

	# Take a single example
	att_point = 50000
	Xinput = Xfull[att_point:att_point+history+2]
	Xwindow_src, Ywindow_src = event_detector.transform_to_window_data(Xinput, Xinput)
	Ypred = event_detector.predict(Xwindow_src)
	benign_errors = (Ypred - Ywindow_src)**2
	print(f'Benign MSE: {np.mean(benign_errors)}')

	adv_ml = False

	for method in exp_methods:

		use_wbox = True
		baseline = build_baseline(Xfull, history, method=method)

		if method == 'SM':
			expl = smooth_grad_mse_explainer.SaliencyMapMseHistoryExplainer()
		
		elif method == 'SG':
			expl = smooth_grad_mse_explainer.SmoothGradMseHistoryExplainer()
		
		elif method == 'IG':
			expl = integrated_gradients_mse_explainer.IntegratedGradientsMseHistoryExplainer()
		
		elif method == 'EG':
			expl = expected_gradients_mse_explainer.ExpectedGradientsMseHistoryExplainer()
	
		elif method == 'LIME':
			expl = lime.lime_tabular.RecurrentTabularExplainer(baseline,
									feature_names=np.arange(baseline.shape[2]),
									verbose=False,
									mode='regression')
			use_wbox = False

		elif method == 'SHAP':
			expl = shap.DeepExplainer(event_detector.inner, baseline)
			use_wbox = False

		else:
			
			use_wbox = False
			expl = None

		if use_wbox:
		
			### Zero test
			expl.setup_explainer(event_detector.inner, Ypred)
			gif_output_zero = expl.explain(Xwindow_src, baselines=Xwindow_src, multiply_by_input=True)
			print(f'Zero test: {np.sum(gif_output_zero)}')
			expl.setup_explainer(event_detector.inner, Ywindow_src)
		
		if adv_ml:
		
			lookup_name = f'{model_name}-{run_name}-{dataset_name}'
			attacks, labels = attack_utils.get_attack_indices(dataset_name)
			
			attack_cols = {
				'SWAT': 16,
				'WADI': 1,
				'TEP': 10,
			}

			run_adv_ml_unit_test(event_detector, lookup_name, attack_cols[dataset_name], expl=expl)
		
		else:

			mag = 2
			mse_ranks, mse_props, exp_ranks, exp_props = run_unit_test(event_detector, expl, Xwindow_src, Ywindow_src, 
				sensor_cols, dataset_name,
				mag=mag, baselines=baseline, verbose=True, method=method)

			print(f'Mag {mag} MSE: avg rank {np.mean(mse_ranks)} avg prop {np.mean(mse_props)}')
			print(f'Mag {mag} EXP: avg rank {np.mean(exp_ranks)} avg prop {np.mean(exp_props)}')

			np.save(f'benchmark-{model_name}-{run_name}-{method}.npy', np.vstack([mse_ranks, mse_props, exp_ranks, exp_props]))

	print("Finished!")
