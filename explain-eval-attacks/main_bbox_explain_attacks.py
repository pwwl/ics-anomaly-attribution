
import lime
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import shap

# Internal imports
import os
import sys
sys.path.append('..')

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from sklearn.model_selection import train_test_split
from data_loader import load_train_data, load_test_data
from main_train import load_saved_model

from live_explainer.score_generator import counterfactual_score_generator, counterfactual_minus_score_generator
from live_explainer.score_generator import mse_score_generator, mse_sd_score_generator
from live_explainer.score_generator import lime_score_generator, shap_score_generator, lemna_score_generator

from tep_utils import idx_to_sen

import attack_utils
import tep_utils
import utils

np.set_printoptions(suppress=True)

HOUR = 2000

def simulate_graph_attack_scoring(event_detector, attacks, Xval, Xtest, Ytest=None, use_per_feature_mse=False, quant=0.9995, window=1, num_samples=1):

	full_val_errors = event_detector.reconstruction_errors(Xval, batches=True)
	full_test_errors = event_detector.reconstruction_errors(Xtest, batches=True)

	n_sensor = Xtest.shape[1]
	history = event_detector.params['history']

	feature_thresholds = np.zeros(n_sensor)
	feature_detect = np.zeros_like(full_test_errors)
	feature_detect_blocklist_idx = []

	## Use per-feature detection
	if use_per_feature_mse:
		for i in range(n_sensor):

			feature_thresholds[i] = np.quantile(full_val_errors[:, i], quant)

			if i in feature_detect_blocklist_idx:
				continue
			feature_detect[:, i] = full_test_errors[:, i] > feature_thresholds[i]

		full_detection = np.sum(feature_detect, axis=1) > 0

	## Use traditional detection
	else:
		detection_cutoff = np.quantile(np.mean(full_val_errors, axis=1), quant)
		full_detection = np.mean(full_test_errors, axis=1) > detection_cutoff

	mserank_results = dict()
	use_counterfact = False
	use_counterfact_minus = False
	use_lime = False
	use_shap = False
	use_lemna = True

	if use_lime:
		Xval_window, _ = event_detector.transform_to_window_data(Xval, Xval)
		shuffle_idx = np.random.permutation(len(Xval_window))[:1000]
		Xval_window_shuf = Xval_window[shuffle_idx]

		lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(Xval_window_shuf,
								feature_names=np.arange(Xval_window_shuf.shape[2]),
								verbose=False,
								mode='regression')

	if use_shap:

		Xval_window, _ = event_detector.transform_to_window_data(Xval, Xval)
		shuffle_idx = np.random.permutation(len(Xval_window))[:1000]
		Xval_window_shuf = Xval_window[shuffle_idx]

		shap_explainer = shap.DeepExplainer(event_detector.inner, Xval_window_shuf)

	# If needed to examine raw values
	# scaler = pickle.load(open(f'models/SWAT-CLEAN_scaler.pkl', "rb"))

	for attack_idx in range(len(attacks)):

		att_start = attacks[attack_idx][0] - history
		att_end = attacks[attack_idx][-1] - history

		print('============================')
		print(f'For attack {attack_idx} window={window} use_per_feature_mse={use_per_feature_mse}')

		## Explore within attack
		attack_detection_idxs = np.where(full_detection[att_start:att_end])[0]
		window_attack_detection_idxs = np.where(np.convolve(full_detection[att_start:att_end], np.ones(window), 'same') // window)[0]

		explain_choice = []
		attack_quant = np.zeros((n_sensor, num_samples))

		if len(window_attack_detection_idxs) > 0:

			# Use a window-based smoothing for scoring
			detect_idx = np.min(window_attack_detection_idxs) + att_start

			# Include the "num_samples" timesteps after the first detection (even if not a TP)
			for time_idx in range(num_samples):

				capture_idx = detect_idx + time_idx
				# print(f'Collecting idx={time_idx} at {capture_idx}')

				if use_counterfact:

					attributions = counterfactual_score_generator(event_detector, capture_idx, Xtest, None)

					# Boost up attributions to cause min value = 0. Okay to do?
					attributions = attributions + np.abs(np.min(attributions))
					attack_quant[:, time_idx] = attributions

				elif use_counterfact_minus:

					attributions = counterfactual_minus_score_generator(event_detector, capture_idx, Xtest, None)

					# Boost up attributions to cause min value = 0. Okay to do?
					attributions = attributions + np.abs(np.min(attributions))
					attack_quant[:, time_idx] = attributions

				# TODO: Expand LIME and SHAP to include averaging
				elif use_lime:
					attack_quant = lime_score_generator(event_detector, lime_explainer, capture_idx, Xtest)

				elif use_shap:
					attack_quant = shap_score_generator(event_detector, shap_explainer, capture_idx, Xtest)

				elif use_lemna:
					sample_quant = lemna_score_generator(event_detector, capture_idx, Xtest)

					# Lemna SG gives a [history, sensors_output]
					attack_quant[:, time_idx] = np.sum(np.abs(sample_quant), axis=0)

				elif use_per_feature_mse:
					attack_quant[:, time_idx] = mse_sd_score_generator(event_detector, capture_idx, Xtest, full_val_errors)

				else:
					attack_quant[:, time_idx] = mse_score_generator(event_detector, capture_idx, Xtest)

			si = np.argmax(np.mean(attack_quant, axis=1))
			explain_choice = [(idx_to_sen(si), np.abs(np.mean(attack_quant, axis=1))[si])]
			mserank_results[f'{attack_idx}_in'] = explain_choice
			mserank_results[f'{attack_idx}_quant_in'] = attack_quant

			print(f'Total: {np.sum(attack_quant)}')

			# Can be used to generate "attack reports"
			# tep_utils.quant_to_sample_ui(attack_quant[:, 0], scaler.inverse_transform(Xtest[detect_idx]), sensor_cols, f'swat-attack-{attack_idx}')

		else:
			print(f'Attack {attack_idx} was completely missed!!')
			mserank_results[f'{attack_idx}_in'] = list()

	return mserank_results

def explain_true_position(event_detector, lookup_name, attacks, Xtest, method='MSE', expl=None, num_samples=1):

	#full_test_errors = event_detector.reconstruction_errors(Xtest, batches=True, verbose=0)

	history = event_detector.params['history']
	nsensors = Xtest.shape[1]

	for attack_idx in range(len(attacks)):

		if method in ['LIME', 'SHAP', 'LEMNA']:
			full_scores = np.zeros((num_samples, history, nsensors))
		else:
			full_scores = np.zeros((num_samples, nsensors))

		# TODO: modified to start from history for now here
		# att_start = attacks[attack_idx][0] + history
		att_start = attacks[attack_idx][0]

		print('============================')

		for i in range(num_samples):

			capture_idx = att_start + i
			capture_start = capture_idx - history - 1

			Xinput = np.expand_dims(Xtest[capture_start:capture_start+history], axis=0)
			Yinput = np.expand_dims(Xtest[capture_idx], axis=0)

			print(f'For attack {attack_idx}: capture {att_start} + {i}')

			if method == 'LEMNA':
				exp_output = lemna_score_generator(event_detector, Xinput, Yinput)
			elif method == 'SHAP':
				exp_output = shap_score_generator(event_detector, expl, Xinput, Yinput)
			else:
				exp_output = (event_detector.predict(Xinput) - Yinput)**2

			full_scores[i] = exp_output

		pickle.dump(full_scores, open(f'explanations-{method}-{lookup_name}-{attack_idx}-true{num_samples}.pkl', 'wb'))

	return

def explain_detect(event_detector, lookup_name, attacks, Xtest, detection_points, method='MSE', expl=None, num_samples=1):

	#full_test_errors = event_detector.reconstruction_errors(Xtest, batches=True, verbose=0)

	history = event_detector.params['history']
	nsensors = Xtest.shape[1]

	for attack_idx in range(len(attacks)):

		if attack_idx in detection_points:

			if method in ['LIME', 'SHAP', 'LEMNA']:
				full_scores = np.zeros((num_samples, history, nsensors))
			else:
				full_scores = np.zeros((num_samples, nsensors))

			# TODO: modified to start from beginning, since detect_idx accounts for history
			att_start = attacks[attack_idx][0]
			detect_idx = detection_points[attack_idx]

			print('============================')

			for i in range(num_samples):

				capture_idx = att_start + detect_idx + i
				capture_start = capture_idx - history - 1

				Xinput = np.expand_dims(Xtest[capture_start:capture_start+history], axis=0)
				Yinput = np.expand_dims(Xtest[capture_idx], axis=0)

				print(f'For attack {attack_idx}: capture {att_start} + {detect_idx + i}')

				if method == 'LEMNA':
					exp_output = lemna_score_generator(event_detector, Xinput, Yinput)
				elif method == 'SHAP':
					exp_output = shap_score_generator(event_detector, expl, Xinput, Yinput)
				else:
					exp_output = (event_detector.predict(Xinput) - Yinput)**2

				full_scores[i] = exp_output

			pickle.dump(full_scores, open(f'explanations-{method}-{lookup_name}-{attack_idx}-detect{num_samples}.pkl', 'wb'))
		
		else:

			print(f'Attack {attack_idx} was missed')

	return

def parse_arguments():

	parser = utils.get_argparser()

	parser.add_argument("--explain_params_methods",
        choices=['MSE', 'LIME', 'SHAP', 'LEMNA'],
        default='AE')

	return parser.parse_args()

if __name__ == "__main__":

	import os
	os.chdir('..')
	args = parse_arguments()
	model_type = args.model
	dataset_name = args.dataset
	exp_method = args.explain_params_methods

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

	run_name = args.run_name
	config = {} # type: Dict[str, str]
	utils.update_config_model(args, config, model_type, dataset_name)
	model_name = config['name']
	event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')
	history = event_detector.params['history']
	attacks, labels = attack_utils.get_attack_indices(dataset_name)

	Xtest, _, _ = load_test_data(dataset_name)
	lookup_name = f'{model_name}-{run_name}'
	num_samples = 150

	if exp_method == 'SHAP':
		Xfull, sensor_cols = load_train_data(dataset_name)
		baseline = utils.build_baseline(Xfull, history, method=exp_method)
		expl = shap.DeepExplainer(event_detector.inner, baseline)
	else:
		expl = None

	# Practical detection
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
	model_detection_points = detection_points[lookup_name]
	explain_detect(event_detector, lookup_name, attacks, Xtest,
			model_detection_points,
			method=exp_method,
			expl=expl,
			num_samples=num_samples)

	# Ideal detection
	explain_true_position(event_detector, lookup_name, attacks, Xtest,
			method=exp_method,
			expl=expl,
			num_samples=num_samples)

	print('All done!')
