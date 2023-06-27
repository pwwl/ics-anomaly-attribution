import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pdb
import pickle

import sys
sys.path.append('explain-eval-manipulations')

from data_loader import load_train_data, load_test_data
from main_train import load_saved_model

from attack_utils import get_attack_indices, get_attack_sds, get_rel_scores, is_actuator
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list, scores_to_rank

import data_loader
import tep_utils
import utils

np.set_printoptions(suppress=True)
DEFAULT_CMAP = plt.get_cmap('Reds', 5)

HOUR = 2000
SCALE = 1

def make_detect_plot_obj_tep():

	models = ['CNN', 'GRU', 'LSTM']
	dataset = 'TEP'
	history = 50
	all_dfs = []
	all_plots = []

	use_corr = True
	run_name = 'results_ns1'

	print(f'Processing {dataset} dataset')
	attacks = tep_utils.get_footer_list(patterns=['cons'])
	sensor_cols = tep_utils.get_short_colnames()

	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	for model in models:

		if model == 'CNN':
			lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

		model_detection_points = detection_points[lookup_name]
		
		scatter_obj = np.zeros((len(attacks), 6))
		first_scatter_obj = np.zeros((len(attacks), 4))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_types_list = []
		pattern_list = []
		detect_list = []

		for foot_idx in range(len(attacks)):
			
			attack_footer = attacks[foot_idx]
			
			splits = attack_footer.split("_")
			pattern = splits[0]
			mag = splits[1]
			label = splits[2]
			col_idx = tep_utils.sen_to_idx(label)
			is_multi = 'solo'
			sd = int(mag[1])
			att_start = 10000

			if attack_footer not in model_detection_points:
				continue

			all_mses = np.load(f'explanations-dir/explain23-tep-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			detect_idx = model_detection_points[attack_footer]
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 

			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			# Using first detection point
			first_mses = all_mses[att_start+history+detect_idx]
			first_sm = smap_scores[0]
			first_shap = shap_scores[0]
			first_lemna = lemna_scores[0]

			first_ranking = scores_to_rank(first_mses, col_idx)
			first_sm_ranking = scores_to_rank(first_sm, col_idx)
			first_shap_ranking = scores_to_rank(first_shap, col_idx)
			first_lemna_ranking = scores_to_rank(first_lemna, col_idx)
			
			scatter_obj[foot_idx, 0] = sd

			first_scatter_obj[foot_idx, 0] = first_ranking
			first_scatter_obj[foot_idx, 1] = first_sm_ranking
			first_scatter_obj[foot_idx, 2] = first_shap_ranking
			first_scatter_obj[foot_idx, 3] = first_lemna_ranking

			print(f'Attack {attack_footer}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

			if label[0] == 's':
				sensor_types_list.append('Sensor')
			elif label[0] == 'a':
				sensor_types_list.append('Actuator')
			else:
				print('Missing label type')
				pdb.set_trace()

			detect_list.append(detect_idx)
			labels_list.append(label)
			multi_list.append(is_multi)
			pattern_list.append(pattern)
			val_types_list.append('float')

		det_idx = first_scatter_obj[:,0] > 0

		print('------------------------')
		print(f'Average first detect MSE ranking: {np.mean(first_scatter_obj[det_idx,0])}')
		print(f'Average first detect sm ranking: {np.mean(first_scatter_obj[det_idx,1])}')
		print(f'Average first detect SHAP ranking: {np.mean(first_scatter_obj[det_idx,2])}')
		print(f'Average first detect LEMNA ranking: {np.mean(first_scatter_obj[det_idx,3])}')
		print('------------------------')

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_types_list,
			'sd_type': pattern_list,
			'sd': scatter_obj[det_idx,0],
			'is_multi': multi_list,
			'detect_point': detect_list,
			'mse_ranking': first_scatter_obj[det_idx,0],
			'sm_ranking': first_scatter_obj[det_idx,1],
			'shap_ranking': first_scatter_obj[det_idx,2],
			'lemna_ranking': first_scatter_obj[det_idx,3],
		})

		pdb.set_trace()

		pickle.dump(df, open(f'realdet-{lookup_name}.pkl', 'wb'))

def make_tep_ideal_plot_obj():

	history = 50
	run_name = 'results_ns1'
	dataset = 'TEP'
	models = ['CNN', 'GRU', 'LSTM']

	print(f'Processing {dataset} dataset')
	attack_footers = tep_utils.get_footer_list(patterns=['cons'])
	sensor_cols = tep_utils.get_short_colnames()

	for model in models:

		if model == 'CNN':
			lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

		scatter_obj = np.zeros((len(attack_footers), 6))
		first_scatter_obj = np.zeros((len(attack_footers), 5))

		labels_list = []
		sensor_types_list = []
		multi_list = []
		pattern_list = []
		sd_list = []
		val_type_list = []

		total_lemna = 0

		for foot_idx in range(len(attack_footers)):
			
			attack_footer = attack_footers[foot_idx]
			splits = attack_footer.split("_")
			
			pattern = splits[0]
			mag = splits[1]
			label = splits[2]
					
			col_idx = tep_utils.sen_to_idx(label)
			att_start = 10000 - history - 1
			att_end = 14000 - history - 1

			print(attack_footer)
			all_mses = np.load(f'explanations-dir/explain23-tep-mses/mses-{lookup_name}-{attack_footer}-ns.npy')

			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-true5.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-true5.pkl', 'rb')) 
			
			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			avg_mses = np.mean(all_mses[att_start+50:att_start+150], axis=0)
			avg_smap = np.mean(smap_scores[51:], axis=0)
			avg150_smap = np.mean(smap_scores, axis=0)
			pre_mses = all_mses[att_start]
			pre_sm = smap_scores[0]

			# Ignoring detections
			first_mses = all_mses[att_start+history]
			first_sm = smap_scores[51]
			first_shap = shap_scores[0]
			first_lemna = lemna_scores[0]
			total_lemna += np.mean(first_lemna)

			first_ranking = scores_to_rank(first_mses, col_idx)
			first_sm_ranking = scores_to_rank(first_sm, col_idx)
			first_shap_ranking = scores_to_rank(first_shap, col_idx)
			first_lemna_ranking = scores_to_rank(first_lemna, col_idx)

			scatter_obj[foot_idx, 0] = int(mag[1])
			scatter_obj[foot_idx, 1] = scores_to_rank(avg_mses, col_idx)
			scatter_obj[foot_idx, 2] = scores_to_rank(avg_smap, col_idx)
			scatter_obj[foot_idx, 3] = scores_to_rank(pre_mses, col_idx)
			scatter_obj[foot_idx, 4] = scores_to_rank(pre_sm, col_idx)
			scatter_obj[foot_idx, 5] = scores_to_rank(avg150_smap, col_idx)

			first_scatter_obj[foot_idx, 0] = first_ranking
			first_scatter_obj[foot_idx, 1] = first_sm_ranking
			first_scatter_obj[foot_idx, 2] = first_shap_ranking
			first_scatter_obj[foot_idx, 3] = first_lemna_ranking

			print(f'Attack {attack_footer}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

			if label[0] == 's':
				sensor_types_list.append('Sensor')
			elif label[0] == 'a':
				sensor_types_list.append('Actuator')
			else:
				print('Missing label type')
				pdb.set_trace()

			labels_list.append(label)
			pattern_list.append(pattern)
			sd_list.append(int(mag[1]))
			multi_list.append('solo')
			val_type_list.append('float')

		print('------------------------')
		print(f'Average first MSE ranking: {np.mean(first_scatter_obj[:,0])}')
		print(f'Average first sm ranking: {np.mean(first_scatter_obj[:,1])}')
		print(f'Average first SHAP ranking: {np.mean(first_scatter_obj[:,2])}')
		print(f'Average first LEMNA ranking: {np.mean(first_scatter_obj[:,3])}')
		print('------------------------')
		print(f'Avg100 ranking: {np.mean(scatter_obj[:,1])}')
		print(f'Avg100 sm ranking: {np.mean(scatter_obj[:,2])}')
		print(f'Avg150 sm ranking: {np.mean(scatter_obj[:,5])}')
		print(f'Pre MSE ranking: {np.mean(scatter_obj[:,3])}')
		print(f'Pre sm ranking: {np.mean(scatter_obj[:,4])}')
		print('------------------------')

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_type_list,
			'sd_type': pattern_list,
			'sd': scatter_obj[:,0],
			'is_multi': multi_list,
			'mse_ranking': first_scatter_obj[:,0],
			'smap_ranking': first_scatter_obj[:,1],
			'shap_ranking': first_scatter_obj[:,2],
			'lemna_ranking': first_scatter_obj[:,3],
			'avg100_mse_ranking': scatter_obj[:,1],
			'avg100_smap_ranking': scatter_obj[:,2],
			'avg150_smap_ranking': scatter_obj[:,5],
			'pre_mse_ranking': scatter_obj[:,3],
			'pre_smap_ranking': scatter_obj[:,4],
		})

		sen_idx = np.where(df['sensor_type'] == 'Sensor')[0]
		act_idx = np.where(df['sensor_type'] == 'Actuator')[0]

		print('------------------------')
		print(f'MSE sen vs act: {np.mean(first_scatter_obj[sen_idx, 0])} vs {np.mean(first_scatter_obj[act_idx, 0])}')
		print(f'SM sen vs act: {np.mean(first_scatter_obj[sen_idx, 1])} vs {np.mean(first_scatter_obj[act_idx, 1])}')
		print(f'SHAP sen vs act: {np.mean(first_scatter_obj[sen_idx, 2])} vs {np.mean(first_scatter_obj[act_idx, 2])}')
		print(f'LEMNA sen vs act: {np.mean(first_scatter_obj[sen_idx, 3])} vs {np.mean(first_scatter_obj[act_idx, 3])}')
		print('------------------------')

		pickle.dump(df, open(f'idealdet-{lookup_name}.pkl', 'wb'))

def make_tep_timing_plot_obj():

	history = 50
	run_name = 'results_ns1'
	dataset = 'TEP'
	model = 'CNN'

	if model == 'CNN':
		lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
	else:
		lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

	print(f'Processing timing for {dataset} dataset')
	
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
	model_detection_points = detection_points[lookup_name]

	all_detection_points = pickle.load(open('ccs-storage/all-detection-points.pkl', 'rb'))
	model_all_detection_points = all_detection_points[lookup_name]

	attack_footers = tep_utils.get_footer_list(patterns=['cons'])
	#attack_footers = tep_utils.get_footer_list()

	sensor_cols = tep_utils.get_short_colnames()
	ncols = len(sensor_cols)

	scatter_obj = np.zeros((len(attack_footers), 17))

	labels_list = []
	sensor_types_list = []
	multi_list = []
	pattern_list = []
	sd_list = []
	val_type_list = []
	detect_point_list = []
	length_list = []
	full_slice_values = np.zeros((len(attack_footers), 150, ncols, 4))

	for foot_idx in range(len(attack_footers)):
		
		attack_footer = attack_footers[foot_idx]
		splits = attack_footer.split("_")
		
		pattern = splits[0]
		mag = splits[1]
		label = splits[2]
				
		col_idx = tep_utils.sen_to_idx(label)
		att_start = 10000 - history - 1
		att_end = 14000 - history - 1

		all_mses = np.load(f'explanations-dir/explain23-tep-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
		smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
		shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
		lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 

		# TODO: if needed, try a different slicing
		smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
		shap_scores = np.sum(np.abs(smap_scores_full), axis=1)
		lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

		mse_rankings = np.zeros(150)
		sm_rankings = np.zeros(150)
		shap_rankings = np.zeros(150)
		lemna_rankings = np.zeros(150)
		slice_avg_rankings = np.zeros(150)
		mse_scores = all_mses[att_start:att_start+150]
		
		mse_tavg = np.zeros(len(sensor_cols))
		sm_tavg = np.zeros(len(sensor_cols))
		shap_tavg = np.zeros(len(sensor_cols))
		lemna_tavg = np.zeros(len(sensor_cols))
		slice_tavg = np.zeros(len(sensor_cols))

		# Ignoring detections
		for i in range(150):

			mse_slice = mse_scores[i]
			sm_slice = smap_scores[i]
			shap_slice = shap_scores[i]
			lemna_slice = lemna_scores[i]

			mse_rankings[i] = scores_to_rank(mse_slice, col_idx)
			sm_rankings[i] = scores_to_rank(sm_slice, col_idx)
			shap_rankings[i] = scores_to_rank(shap_slice, col_idx)
			lemna_rankings[i] = scores_to_rank(lemna_slice, col_idx)

			mse_slice_norm = mse_slice / np.sum(mse_slice)
			sm_slice_norm = sm_slice / np.sum(sm_slice)
			shap_slice_norm = shap_slice / np.sum(shap_slice)
			lemna_slice_norm = lemna_slice / np.sum(lemna_slice)

			full_slice_values[foot_idx, i, :, 0] = mse_slice_norm
			full_slice_values[foot_idx, i, :, 1] = sm_slice_norm
			full_slice_values[foot_idx, i, :, 2] = shap_slice_norm
			full_slice_values[foot_idx, i, :, 3] = lemna_slice_norm

			slice_avg = np.sum(np.vstack([mse_slice_norm, sm_slice_norm, lemna_slice_norm]), axis=0)
			slice_avg_rankings[i] = scores_to_rank(slice_avg, col_idx)

			mse_tavg += mse_slice_norm
			sm_tavg += sm_slice_norm
			shap_tavg += shap_slice_norm
			lemna_tavg += lemna_slice_norm
			slice_tavg += slice_avg

		mse_avg_ranking = scores_to_rank(mse_tavg, col_idx)
		sm_avg_ranking = scores_to_rank(sm_tavg, col_idx)
		shap_avg_ranking = scores_to_rank(shap_tavg, col_idx)
		lemna_avg_ranking = scores_to_rank(lemna_tavg, col_idx)
		avg_avg_ranking = scores_to_rank(slice_tavg, col_idx)

		##################################################################

		if attack_footer in model_detection_points:
			
			detect_point = model_detection_points[attack_footer]
			all_points = model_all_detection_points[attack_footer]
			
			window = 5
			binary_signal = np.zeros(np.max(all_points) + 1)
			binary_signal[all_points] = 1
			detection = np.convolve(binary_signal, np.ones(window), 'same') // window

			if np.sum(detection) > 0:
				new_detect_point = np.min(np.where(detection)) - 2

				if detect_point != new_detect_point:
					print(f'Attack {attack_footer} after windowing: detect point moved from {detect_point} to {new_detect_point}')
				else:
					print(f'Attack {attack_footer} same detect point {detect_point}.')

			else:
				print(f'Attack {attack_footer} is now missed.')
		
		else:

			print(f'Attack {attack_footer} still missed.')
			detect_point = -1

		##################################################################

		scatter_obj[foot_idx, 0] = int(mag[1])
		scatter_obj[foot_idx, 1] = np.mean(mse_rankings)
		scatter_obj[foot_idx, 2] = np.min(mse_rankings)
		scatter_obj[foot_idx, 3] = mse_avg_ranking

		scatter_obj[foot_idx, 4] = np.mean(sm_rankings)
		scatter_obj[foot_idx, 5] = np.min(sm_rankings)
		scatter_obj[foot_idx, 6] = sm_avg_ranking

		scatter_obj[foot_idx, 7] = np.mean(shap_rankings)
		scatter_obj[foot_idx, 8] = np.min(shap_rankings)
		scatter_obj[foot_idx, 9] = shap_avg_ranking

		scatter_obj[foot_idx, 10] = np.mean(lemna_rankings)
		scatter_obj[foot_idx, 11] = np.min(lemna_rankings)
		scatter_obj[foot_idx, 12] = lemna_avg_ranking

		scatter_obj[foot_idx, 13] = np.mean(slice_avg_rankings)
		scatter_obj[foot_idx, 14] = np.min(slice_avg_rankings)
		scatter_obj[foot_idx, 15] = avg_avg_ranking

		if label[0] == 's':
			sensor_types_list.append('Sensor')
		elif label[0] == 'a':
			sensor_types_list.append('Actuator')
		else:
			print('Missing label type')
			pdb.set_trace()

		detect_point_list.append(detect_point)
		labels_list.append(label)
		pattern_list.append(pattern)
		sd_list.append(int(mag[1]))
		multi_list.append('solo')
		val_type_list.append('float')
		length_list.append(att_end - att_start)

	df = pd.DataFrame({
		'sensor': labels_list,
		'sensor_type': sensor_types_list,
		'val_type': val_type_list,
		'sd_type': pattern_list,
		'sd': scatter_obj[:,0],
		'is_multi': multi_list,
		'attack_len': length_list,
		'detect_point': detect_point_list,
		'mse_avg_ranking': scatter_obj[:, 1],
		'mse_best_ranking': scatter_obj[:, 2],
		'mse_tavg_ranking': scatter_obj[:, 3],
		'sm_avg_ranking': scatter_obj[:, 4],
		'sm_best_ranking': scatter_obj[:, 5],
		'sm_tavg_ranking': scatter_obj[:, 6],
		'shap_avg_ranking': scatter_obj[:, 7],
		'shap_best_ranking': scatter_obj[:, 8],
		'shap_tavg_ranking': scatter_obj[:, 9],
		'lemna_avg_ranking': scatter_obj[:, 10],
		'lemna_best_ranking': scatter_obj[:, 11],
		'lemna_tavg_ranking': scatter_obj[:, 12],
		'slice_avg_ranking': scatter_obj[:, 13],
		'slice_best_ranking': scatter_obj[:, 14],
		'slice_tavg_ranking': scatter_obj[:, 15],
		'slice_tavg_opt_ranking': scatter_obj[:, 16],
	})

	pdb.set_trace()

	pickle.dump(df, open(f'timing-{lookup_name}.pkl', 'wb'))
	pickle.dump(full_slice_values, open(f'full-values-{lookup_name}.pkl', 'wb'))
	

def make_detect_timing():

	models = ['CNN', 'GRU', 'LSTM']
	dataset = 'TEP'
	history = 50
	all_dfs = []
	all_plots = []

	run_name = 'results_ns1'

	print(f'Processing {dataset} dataset')
	attacks = tep_utils.get_footer_list(patterns=['cons'])
	sensor_cols = tep_utils.get_short_colnames()
	ncols = len(sensor_cols)

	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	for model in models:

		if model == 'CNN':
			lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

		model_detection_points = detection_points[lookup_name]
		
		scatter_obj = np.zeros((len(attacks), 16))

		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_types_list = []
		pattern_list = []
		detect_list = []
		length_list = []
		full_slice_values = np.zeros((len(attacks), 150, ncols, 4))

		for foot_idx in range(len(attacks)):
			
			attack_footer = attacks[foot_idx]
			
			splits = attack_footer.split("_")
			pattern = splits[0]
			mag = splits[1]
			label = splits[2]
			col_idx = tep_utils.sen_to_idx(label)
			is_multi = 'solo'
			sd = int(mag[1])
			att_start = 10000

			if attack_footer not in model_detection_points:
				continue

			all_mses = np.load(f'explanations-dir/explain23-tep-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			detect_idx = model_detection_points[attack_footer]
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 

			capture_start = att_start + history + detect_idx
			mse_scores = all_mses[capture_start:capture_start+150]
			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			mse_rankings = np.zeros(150)
			sm_rankings = np.zeros(150)
			shap_rankings = np.zeros(150)
			lemna_rankings = np.zeros(150)
			slice_avg_rankings = np.zeros(150)
			
			mse_tavg = np.zeros(len(sensor_cols))
			sm_tavg = np.zeros(len(sensor_cols))
			shap_tavg = np.zeros(len(sensor_cols))
			lemna_tavg = np.zeros(len(sensor_cols))
			slice_tavg = np.zeros(len(sensor_cols))

			# Averaging from detected point
			for i in range(150):

				mse_slice = mse_scores[i]
				sm_slice = smap_scores[i]
				shap_slice = shap_scores[i]
				lemna_slice = lemna_scores[i]

				mse_rankings[i] = scores_to_rank(mse_slice, col_idx)
				sm_rankings[i] = scores_to_rank(sm_slice, col_idx)
				shap_rankings[i] = scores_to_rank(shap_slice, col_idx)
				lemna_rankings[i] = scores_to_rank(lemna_slice, col_idx)

				mse_slice_norm = mse_slice / np.sum(mse_slice)
				sm_slice_norm = sm_slice / np.sum(sm_slice)
				shap_slice_norm = shap_slice / np.sum(shap_slice)
				lemna_slice_norm = lemna_slice / np.sum(lemna_slice)

				full_slice_values[foot_idx, i, :, 0] = mse_slice_norm
				full_slice_values[foot_idx, i, :, 1] = sm_slice_norm
				full_slice_values[foot_idx, i, :, 2] = shap_slice_norm
				full_slice_values[foot_idx, i, :, 3] = lemna_slice_norm

				slice_avg = np.sum(np.vstack([mse_slice_norm, sm_slice_norm, lemna_slice_norm]), axis=0)
				slice_avg_rankings[i] = scores_to_rank(slice_avg, col_idx)

				mse_tavg += mse_slice_norm
				sm_tavg += sm_slice_norm
				shap_tavg += shap_slice_norm
				lemna_tavg += lemna_slice_norm
				slice_tavg += slice_avg

			mse_avg_ranking = scores_to_rank(mse_tavg, col_idx)
			sm_avg_ranking = scores_to_rank(sm_tavg, col_idx)
			shap_avg_ranking = scores_to_rank(shap_tavg, col_idx)
			lemna_avg_ranking = scores_to_rank(lemna_tavg, col_idx)
			avg_avg_ranking = scores_to_rank(slice_tavg, col_idx)
			
			scatter_obj[foot_idx, 0] = sd
			scatter_obj[foot_idx, 1] = np.mean(mse_rankings)
			scatter_obj[foot_idx, 2] = np.min(mse_rankings)
			scatter_obj[foot_idx, 3] = mse_avg_ranking

			scatter_obj[foot_idx, 4] = np.mean(sm_rankings)
			scatter_obj[foot_idx, 5] = np.min(sm_rankings)
			scatter_obj[foot_idx, 6] = sm_avg_ranking

			scatter_obj[foot_idx, 7] = np.mean(shap_rankings)
			scatter_obj[foot_idx, 8] = np.min(shap_rankings)
			scatter_obj[foot_idx, 9] = shap_avg_ranking

			scatter_obj[foot_idx, 10] = np.mean(lemna_rankings)
			scatter_obj[foot_idx, 11] = np.min(lemna_rankings)
			scatter_obj[foot_idx, 12] = lemna_avg_ranking

			scatter_obj[foot_idx, 13] = np.mean(slice_avg_rankings)
			scatter_obj[foot_idx, 14] = np.min(slice_avg_rankings)
			scatter_obj[foot_idx, 15] = avg_avg_ranking

			if label[0] == 's':
				sensor_types_list.append('Sensor')
			elif label[0] == 'a':
				sensor_types_list.append('Actuator')
			else:
				print('Missing label type')
				pdb.set_trace()

			length_list.append(4000)
			detect_list.append(detect_idx)
			labels_list.append(label)
			multi_list.append(is_multi)
			pattern_list.append(pattern)
			val_types_list.append('float')

		det_idx = scatter_obj[:,0] > 0

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_types_list,
			'sd_type': pattern_list,
			'sd': scatter_obj[det_idx,0],
			'is_multi': multi_list,
			'attack_len': length_list,
			'detect_point': detect_list,
			'mse_avg_ranking': scatter_obj[det_idx, 1],
			'mse_best_ranking': scatter_obj[det_idx, 2],
			'mse_tavg_ranking': scatter_obj[det_idx, 3],
			'sm_avg_ranking': scatter_obj[det_idx, 4],
			'sm_best_ranking': scatter_obj[det_idx, 5],
			'sm_tavg_ranking': scatter_obj[det_idx, 6],
			'shap_avg_ranking': scatter_obj[det_idx, 7],
			'shap_best_ranking': scatter_obj[det_idx, 8],
			'shap_tavg_ranking': scatter_obj[det_idx, 9],
			'lemna_avg_ranking': scatter_obj[det_idx, 10],
			'lemna_best_ranking': scatter_obj[det_idx, 11],
			'lemna_tavg_ranking': scatter_obj[det_idx, 12],
			'slice_avg_ranking': scatter_obj[det_idx, 13],
			'slice_best_ranking': scatter_obj[det_idx, 14],
			'slice_tavg_ranking': scatter_obj[det_idx, 15],
		})

		pdb.set_trace()

		pickle.dump(df, open(f'real-timing-{lookup_name}.pkl', 'wb'))
		pickle.dump(full_slice_values, open(f'full-values-real-{lookup_name}.pkl', 'wb'))

def make_stealth_plot_obj():

	models = ['CNN', 'GRU', 'LSTM']
	dataset = 'TEP'
	history = 50

	run_name = 'results_ns1'

	for model in models:

		if model == 'CNN':
			lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

		print(f'Processing {dataset} dataset')
		attacks = tep_utils.get_footer_list(patterns=['cons', 'csum', 'line'], mags=['p2s'], locations='pid')

		detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]
		
		scatter_obj = np.zeros((len(attacks), 6))
		first_scatter_obj = np.zeros((len(attacks), 4))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		pattern_list = []
		detect_list = []

		for foot_idx in range(len(attacks)):
			
			attack_footer = attacks[foot_idx]
			
			splits = attack_footer.split("_")
			pattern = splits[0]
			mag = splits[1]
			label = splits[2]
			col_idx = tep_utils.sen_to_idx(label)
			is_multi = 'solo'
			sd = int(mag[1])
			att_start = 10000

			if attack_footer not in model_detection_points:
				continue

			all_mses = np.load(f'explanations-dir/explain23-tep-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			detect_idx = model_detection_points[attack_footer]
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-detect5.pkl', 'rb')) 

			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			# Using first detection point
			first_mses = all_mses[att_start+history+detect_idx]
			first_sm = smap_scores[0]
			first_shap = shap_scores[0]
			first_lemna = lemna_scores[0]

			first_ranking = scores_to_rank(first_mses, col_idx)
			first_sm_ranking = scores_to_rank(first_sm, col_idx)
			first_shap_ranking = scores_to_rank(first_shap, col_idx)
			first_lemna_ranking = scores_to_rank(first_lemna, col_idx)

			scatter_obj[foot_idx, 0] = sd

			first_scatter_obj[foot_idx, 0] = first_ranking
			first_scatter_obj[foot_idx, 1] = first_sm_ranking
			first_scatter_obj[foot_idx, 2] = first_shap_ranking
			first_scatter_obj[foot_idx, 3] = first_lemna_ranking

			if label[0] == 's':
				sensor_types_list.append('Sensor')
			elif label[0] == 'a':
				sensor_types_list.append('Actuator')
			else:
				print('Missing label type')
				pdb.set_trace()

			detect_list.append(detect_idx)
			labels_list.append(label)
			multi_list.append(is_multi)
			pattern_list.append(pattern)

		det_idx = first_scatter_obj[:,0] > 0

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'sd_type': pattern_list,
			'sd': scatter_obj[det_idx,0],
			'is_multi': multi_list,
			'detect_point': detect_list,
			'mse_ranking': first_scatter_obj[det_idx,0],
			'smap_ranking': first_scatter_obj[det_idx,1],
			'shap_ranking': first_scatter_obj[det_idx,2],
			'lemna_ranking': first_scatter_obj[det_idx,3],
		})

		pickle.dump(df, open(f'realdet-stealth-{lookup_name}.pkl', 'wb'))
		
		sensors = ['s2', 's3', 's8', 's14', 's17']	
		expls = ['detect_point', 'mse_ranking', 'smap_ranking', 'shap_ranking', 'lemna_ranking']
		dfs = df[df['sensor'].isin(sensors)]

		for exp in expls:
			
			avg_cons = np.mean(dfs[dfs['sd_type'] == 'cons'][exp])
			avg_csum = np.mean(dfs[dfs['sd_type'] == 'csum'][exp])
			avg_line = np.mean(dfs[dfs['sd_type'] == 'line'][exp])

			print(f'Avg cons {exp}: {avg_cons}')
			print(f'Avg csum {exp}: {avg_csum}')
			print(f'Avg line {exp}: {avg_line}')

		pdb.set_trace()

def parse_arguments():

	parser = utils.get_argparser()
	return parser.parse_args()

if __name__ == "__main__":

	make_tep_ideal_plot_obj()
	make_detect_plot_obj_tep()

	make_tep_timing_plot_obj()
	make_detect_timing()

	make_stealth_plot_obj()

	print('Done')
