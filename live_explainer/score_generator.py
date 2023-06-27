#### A collection of various score generator strategies.

import numpy as np
import pdb
import tep_utils

from pygflasso import MultiTaskGFLasso

def mse_score_generator(event_detector, selected_index, Xtest):

	history = event_detector.params['history']
	
	# Go at least 1 history deep in the history
	region_start = selected_index 
	region_end = selected_index + history + 2

	Xinput = Xtest[region_start : region_end]
	rec_errors = event_detector.reconstruction_errors(Xinput, batches=True)[0]

	return rec_errors

def mse_sd_score_generator(event_detector, selected_index, Xtest, full_val_errors):
	
	history = event_detector.params['history']
	
	# Go at least 1 history deep in the history
	region_start = selected_index 
	region_end = selected_index + history + 2

	Xinput = Xtest[region_start : region_end]
	rec_errors = event_detector.reconstruction_errors(Xinput, batches=True)[0]

	# full_test_errors = event_detector.reconstruction_errors(Xtest, batches=True)

	n_sensor = Xtest.shape[1]
	per_feature_sd = np.zeros(n_sensor)

	for si in range(n_sensor):
		#local_error = rec_errors[selected_index, si]
		local_error = rec_errors[si]
		inv_sd = np.abs(local_error - np.mean(full_val_errors[:, si])) / (np.std(full_val_errors[:, si]) + 1e-3)

		per_feature_sd[si] = inv_sd

	return per_feature_sd

def ci_score_generator(selected_index, Xval, Xtest):

	# TODO: unlike the mse_score_generator, this needs untransformed data. 
	# Safe to assume for the input? Or force-include scalar object to untransform?
	# TODO: rewrite with assumed transform
	val_means = np.mean(Xval, axis=0)
	val_stds = np.std(Xval, axis=0)

	n_sensor = Xtest.shape[1]
	
	inv_stds = np.zeros(n_sensor)
	feature_detect_blocklist_idx = [45, 49, 52]

	for i in range(n_sensor):

		if i in feature_detect_blocklist_idx:
			inv_stds[i] = 0
			continue
		
		inv_stds[si] = np.abs(Xtest[selected_index, si] - val_means[si]) / val_stds[si]

	return inv_stds

def gradient_score_generator(event_detector, selected_index, explainer, Xtest, baseline, use_top_feat=False):

	history = event_detector.params['history']
	
	# Go at least 1 history deep in the history
	region_start = selected_index 
	region_end = selected_index + history + 2

	Xinput = Xtest[region_start : region_end]
	Xinput_src, Yinput_src = event_detector.transform_to_window_data(Xinput, Xinput)

	rec_errors = event_detector.reconstruction_errors(Xinput, batches=False)

	if use_top_feat:

		top_feat = np.argmax(rec_errors)

		# Only the first true positive is taken
		explainer.setup_explainer(event_detector.inner, Yinput_src, top_feat)
		explain_output = explainer.explain(Xinput_src, baselines=baseline, multiply_by_input=True)

	else:

		# Only the first true positive is taken
		explainer.setup_explainer(event_detector.inner, Yinput_src)
		explain_output = explainer.explain(Xinput_src, baselines=baseline, multiply_by_input=True)

	return explain_output

def counterfactual_score_generator(event_detector, Xinput, Yinput, baseline=None):
	
	history = event_detector.params['history']

	# Find baseline, use 0s as default
	if baseline is None:
		baseline = 0 * Xinput

	Ybase = event_detector.predict(baseline)
	baseline_mse = np.mean((Ybase - event_detector.predict(baseline)) ** 2)
	
	n_sensors = Ybase.shape[1]
	attributions = np.zeros(n_sensors)

	# Replace each feature with baseline
	for i in range(n_sensors):

		Xcounterfact = baseline.copy()
		Xcounterfact[:, :, i] = Xinput[:, :, i]

		new_mse = np.mean((Ybase - event_detector.predict(Xcounterfact)) ** 2)

		# Track the increase in MSE when replacing baseline feature with its attacked version
		attributions[i] = new_mse - baseline_mse

	return attributions

def counterfactual_minus_score_generator(event_detector, Xinput, Yinput, baseline=None):
	
	history = event_detector.params['history']

	# Find baseline, use 0s as default
	if baseline is None:
		baseline = 0 * Xinput

	input_mse = np.mean((Yinput - event_detector.predict(Xinput)) ** 2)

	n_sensors = Yinput.shape[1]
	attributions = np.zeros(n_sensors)

	# Replace each feature with baseline
	for i in range(n_sensors):

		Xcounterfact = Xinput.copy()
		Xcounterfact[:, :, i] = baseline[:, :, i]

		new_mse = np.mean((Yinput - event_detector.predict(Xcounterfact)) ** 2)

		# Track the drop in MSE when replacing an attacked feature with its baseline
		attributions[i] = new_mse - input_mse

	return attributions

def generate_blackbox_explainer_samples(event_detector, Xinput, Yinput, num_samples=100):

	history = event_detector.params['history']
	n_features = Yinput.shape[1]
	assert (n_features > 1)

	# Pick the feature with the highest error
	Xpert = np.zeros((num_samples, history * n_features))
	Ypert = np.zeros((num_samples, n_features))

	# Perform 100 samples
	for i in range(num_samples):
		
		Xinput_pert = Xinput + (np.random.randn(1, history, n_features) * 0.05)
		#Xinput_pert_src, _ = event_detector.transform_to_window_data(Xinput_pert, Xinput_pert)

		Ypert[i] = (event_detector.predict(Xinput_pert) - Yinput) ** 2
		Xpert[i] = Xinput_pert.flatten()

	return Xpert, Ypert

def lime_score_generator(event_detector, lime_explainer, Xinput, Yinput):

	history = event_detector.params['history']
	n_features = Yinput.shape[1]

	# Pick the feature with the highest error
	rec_errors = (event_detector.predict(Xinput) - Yinput)**2
	topfeature = np.argmax(np.mean(rec_errors, axis=0))

	# If locally sampling examples
	# Xpert, _ = generate_blackbox_explainer_samples(event_detector, Xtest, selected_index, 
	# 	num_samples=n_samples)
	# Xpert = Xpert.reshape(n_samples, history, n_features)

	# ### Perform LIME explanation
	# lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(Xpert,
	# 		feature_names=np.arange(n_features),
	# 		verbose=False,
	# 		mode='regression')

	def predict_func(X):
		return event_detector.predict(X)[:, topfeature]

	lime_out = lime_explainer.explain_instance(Xinput, 
		  predict_func, 
		  num_features=history * n_features)

	explanation_full = np.zeros((history, n_features))

	for i in range(len(lime_out.as_list())):
		
		sensor, history_amt, value = tep_utils.lime_exp_to_feature(lime_out, i)

		if explanation_full[history_amt, sensor] != 0:
			print('Duplicate?')
			pdb.set_trace()
		else:
			explanation_full[history_amt, sensor] = value

	return explanation_full

def shap_score_generator(event_detector, shap_explainer, Xinput, Yinput):

	history = event_detector.params['history']
	n_features = Yinput.shape[1]
	
	# Pick the feature with the highest error
	rec_errors = (event_detector.predict(Xinput) - Yinput)**2
	topfeature = np.argmax(np.mean(rec_errors, axis=0))

	# Xpert, Ypert = generate_blackbox_explainer_samples(event_detector, Xtest, selected_index, 
	# 	num_samples=n_samples)
	# Xpert = Xpert.reshape(n_samples, history, n_features)

	### Perform SHAP explanation
	# shap_explainer = shap.DeepExplainer(event_detector.inner, Xpert)
	
	shap_values = shap_explainer.shap_values(Xinput)

	return shap_values[topfeature][0]

def lemna_score_generator(event_detector, Xinput, Yinput):

	history = event_detector.params['history']
	n_features = Yinput.shape[1]
	assert (n_features > 1)

	# Go at least 1 history deep in the history
	# rregion_start = selected_index - history - 2
	# rregion_end = selected_index

	# Pick the feature with the highest error
	#Xinput = Xtest[region_start : region_end]
	rec_errors = (event_detector.predict(Xinput) - Yinput)**2
	topfeature = np.argmax(np.mean(rec_errors, axis=0))

	Xpert, Ypert = generate_blackbox_explainer_samples(event_detector, Xinput, Yinput,
		num_samples=100)

	### Perform LEMNA explanation
	G = np.identity(n_features)
	gfl = MultiTaskGFLasso(G, verbose=True, max_iter=100, tol=10**-5).fit(Xpert, Ypert)

	total_weights = gfl.coef_.real
	lemna_values = total_weights[topfeature].reshape((history, n_features))

	return lemna_values
