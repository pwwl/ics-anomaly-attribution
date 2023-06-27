import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pdb
import pickle

import sys
sys.path.append('explain-eval-attacks')

from data_loader import load_train_data, load_test_data
from main_train import load_saved_model

from attack_utils import get_attack_indices, get_attack_sds, get_rel_scores, is_actuator
from tep_utils import scores_to_rank

import data_loader
import utils

np.set_printoptions(suppress=True)
DEFAULT_CMAP = plt.get_cmap('Reds', 5)

HOUR = 2000
SCALE = 1

def make_detect_plot_obj():

	models = ['CNN']
	datasets = ['SWAT', 'WADI']
	history = 50
	all_dfs = []
	all_plots = []

	run_name = 'results_ns1'

	for dataset in datasets:

		Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)

		for model in models:

			if model == 'CNN':
				lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
			else:
				lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

			print(f'Processing {dataset} dataset')
			
			attacks, labels = get_attack_indices(dataset)
			detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
			model_detection_points = detection_points[lookup_name]
			sds = get_attack_sds(dataset)
			
			all_mses = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
			val_mses = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')

			print(f'for {lookup_name}')
			print(f'avg_val_mse: {np.mean(val_mses)}')

			scatter_obj = np.zeros((len(sds), 6))
			first_scatter_obj = np.zeros((len(sds), 4))
			
			labels_list = []
			multi_list = []
			sensor_types_list = []
			val_types_list = []
			pattern_list = []
			detect_list = []

			for sd_idx in range(len(sds)):
				
				sd_obj = sds[sd_idx]
				atk_idx = sd_obj[0]
				label = sd_obj[1]
				is_multi = sd_obj[3]
				sd = np.abs(sd_obj[4])
				col_idx = sensor_cols.index(label)

				att_start = np.min(attacks[atk_idx])

				if atk_idx not in model_detection_points:
					#print(f'Attack {atk_idx} missed. Skipping..')
					continue

				detect_idx = model_detection_points[atk_idx]
				smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{atk_idx}-detect5.pkl', 'rb')) 
				shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{atk_idx}-detect5.pkl', 'rb')) 
				lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{atk_idx}-detect5.pkl', 'rb')) 

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

				scatter_obj[sd_idx, 0] = sd
				first_scatter_obj[sd_idx, 0] = first_ranking
				first_scatter_obj[sd_idx, 1] = first_sm_ranking
				first_scatter_obj[sd_idx, 2] = first_shap_ranking
				first_scatter_obj[sd_idx, 3] = first_lemna_ranking

				print(f'Attack {sd_obj}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

				if is_actuator(dataset, label):
					sensor_types_list.append('Actuator')
				else:
					sensor_types_list.append('Sensor')

				detect_list.append(detect_idx)
				labels_list.append(label)
				multi_list.append(is_multi)
				pattern_list.append(sd_obj[2])

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
				'lemna_ranking': first_scatter_obj[det_idx,3]
			})

			pdb.set_trace()

			pickle.dump(df, open(f'realdet-{lookup_name}.pkl', 'wb'))

def make_detect_timing():

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI']
	history = 50
	all_dfs = []
	all_plots = []

	run_name = 'results_ns1'

	for dataset in datasets:

		Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)
		ncols = len(sensor_cols)

		for model in models:

			if model == 'CNN':
				lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
			else:
				lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

			print(f'Processing {dataset} dataset')
			
			attacks, labels = get_attack_indices(dataset)
			detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
			model_detection_points = detection_points[lookup_name]
			sds = get_attack_sds(dataset)
			
			all_mses = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
			val_mses = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')

			print(f'for {lookup_name}')
			print(f'avg_val_mse: {np.mean(val_mses)}')

			scatter_obj = np.zeros((len(sds), 16))
			
			labels_list = []
			multi_list = []
			sensor_types_list = []
			val_types_list = []
			pattern_list = []
			detect_list = []
			length_list = []
			full_slice_values = np.zeros((len(sds), 150, ncols, 4))

			for sd_idx in range(len(sds)):
				
				sd_obj = sds[sd_idx]
				atk_idx = sd_obj[0]
				label = sd_obj[1]
				is_multi = sd_obj[3]
				sd = np.abs(sd_obj[4])
				col_idx = sensor_cols.index(label)

				att_start = np.min(attacks[atk_idx])
				att_end = np.max(attacks[atk_idx])

				if atk_idx not in model_detection_points:
					#print(f'Attack {atk_idx} missed. Skipping..')
					continue

				detect_idx = model_detection_points[atk_idx]
				smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{atk_idx}-detect150.pkl', 'rb')) 
				shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{atk_idx}-detect150.pkl', 'rb')) 
				lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{atk_idx}-detect150.pkl', 'rb')) 

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
					shap_slice_norm = sm_slice / np.sum(sm_slice)
					lemna_slice_norm = lemna_slice / np.sum(lemna_slice)

					full_slice_values[sd_idx, i, :, 0] = mse_slice_norm
					full_slice_values[sd_idx, i, :, 1] = sm_slice_norm
					full_slice_values[sd_idx, i, :, 2] = shap_slice_norm
					full_slice_values[sd_idx, i, :, 3] = lemna_slice_norm

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

				scatter_obj[sd_idx, 0] = sd
				scatter_obj[sd_idx, 1] = np.mean(mse_rankings)
				scatter_obj[sd_idx, 2] = np.min(mse_rankings)
				scatter_obj[sd_idx, 3] = mse_avg_ranking

				scatter_obj[sd_idx, 4] = np.mean(sm_rankings)
				scatter_obj[sd_idx, 5] = np.min(sm_rankings)
				scatter_obj[sd_idx, 6] = sm_avg_ranking

				scatter_obj[sd_idx, 7] = np.mean(shap_rankings)
				scatter_obj[sd_idx, 8] = np.min(shap_rankings)
				scatter_obj[sd_idx, 9] = shap_avg_ranking

				scatter_obj[sd_idx, 10] = np.mean(lemna_rankings)
				scatter_obj[sd_idx, 11] = np.min(lemna_rankings)
				scatter_obj[sd_idx, 12] = lemna_avg_ranking

				scatter_obj[sd_idx, 13] = np.mean(slice_avg_rankings)
				scatter_obj[sd_idx, 14] = np.min(slice_avg_rankings)
				scatter_obj[sd_idx, 15] = avg_avg_ranking

				if is_actuator(dataset, label):
					sensor_types_list.append('Actuator')
					val_types_list.append('bool')
				else:
					sensor_types_list.append('Sensor')
					val_types_list.append('float')

				detect_list.append(detect_idx)
				labels_list.append(label)
				multi_list.append(is_multi)
				pattern_list.append(sd_obj[2])
				length_list.append(att_start - att_end)

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

def make_ideal_plot_obj():

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI']
	history = 50
	all_dfs = []
	all_plots = []

	run_name = 'results_ns1'

	for dataset in datasets:

		for model in models:

			if model == 'CNN':
				lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
			else:
				lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

			print(f'Processing {dataset} dataset')
			Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)
			
			attacks, labels = get_attack_indices(dataset)
			sds = get_attack_sds(dataset)
			
			all_mses = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
			val_mses = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')

			print(f'for {lookup_name}')
			print(f'avg_val_mse: {np.mean(val_mses)}')

			scatter_obj = np.zeros((len(sds), 6))
			first_scatter_obj = np.zeros((len(sds), 4))
			
			labels_list = []
			multi_list = []
			sensor_types_list = []
			val_type_list = []
			pattern_list = []

			for sd_idx in range(len(sds)):
				sd_obj = sds[sd_idx]
				atk_idx = sd_obj[0]
				label = sd_obj[1]
				is_multi = sd_obj[3]
				sd = np.abs(sd_obj[4])
				col_idx = sensor_cols.index(label)

				att_start = np.min(attacks[atk_idx]) - history - 1
				att_end = np.max(attacks[atk_idx]) - history - 1

				smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{atk_idx}-true150.pkl', 'rb')) 
				shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{atk_idx}-true5.pkl', 'rb')) 
				lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{atk_idx}-true5.pkl', 'rb')) 

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

				first_ranking = scores_to_rank(first_mses, col_idx)
				first_sm_ranking = scores_to_rank(first_sm, col_idx)
				first_shap_ranking = scores_to_rank(first_shap, col_idx)
				first_lemna_ranking = scores_to_rank(first_lemna, col_idx)

				print(f'Attack {sd_obj}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

				scatter_obj[sd_idx, 0] = sd
				scatter_obj[sd_idx, 1] = scores_to_rank(avg_mses, col_idx)
				scatter_obj[sd_idx, 2] = scores_to_rank(avg_smap, col_idx)
				scatter_obj[sd_idx, 3] = scores_to_rank(pre_mses, col_idx)
				scatter_obj[sd_idx, 4] = scores_to_rank(pre_sm, col_idx)
				scatter_obj[sd_idx, 5] = scores_to_rank(avg150_smap, col_idx)

				first_scatter_obj[sd_idx, 0] = first_ranking
				first_scatter_obj[sd_idx, 1] = first_sm_ranking
				first_scatter_obj[sd_idx, 2] = first_shap_ranking
				first_scatter_obj[sd_idx, 3] = first_lemna_ranking

				if is_actuator(dataset, label):
					sensor_types_list.append('Actuator')
					val_type_list.append('bool')
				else:
					sensor_types_list.append('Sensor')
					val_type_list.append('float')

				labels_list.append(label)
				multi_list.append(is_multi)
				pattern_list.append(sd_obj[2])

			print('------------------------')
			print(f'Average first MSE ranking: {np.mean(first_scatter_obj[:,0])}')
			print(f'Average first sm ranking: {np.mean(first_scatter_obj[:,1])}')
			print(f'Average first SHAP ranking: {np.mean(first_scatter_obj[:,2])}')
			print(f'Average first LEMNA ranking: {np.mean(first_scatter_obj[:,3])}')
			print('------------------------')
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
				'sm_ranking': first_scatter_obj[:,1],
				'shap_ranking': first_scatter_obj[:,2],
				'lemna_ranking': first_scatter_obj[:,3],
				'avg100_mse_ranking': scatter_obj[:,1],
				'avg100_sm_ranking': scatter_obj[:,2],
				'avg150_smap_ranking': scatter_obj[:,5],
				'pre_mse_ranking': scatter_obj[:,3],
				'pre_sm_ranking': scatter_obj[:,4],
			})

			pickle.dump(df, open(f'idealdet-{lookup_name}.pkl', 'wb'))

def make_timing_plot_obj():

	history = 50
	run_name = 'results_ns1'
	datasets = ['SWAT', 'WADI']
	model = 'GRU'
		
	for dataset in datasets:

		_, _, sensor_cols = data_loader.load_test_data(dataset)
		ncols = len(sensor_cols)

		if model == 'CNN':
			lookup_name = f'CNN-{dataset}-l2-hist50-kern3-units64-{run_name}'
		else:
			lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

		print(f'Processing {dataset} dataset')
		attacks, labels = get_attack_indices(dataset)
		sds = get_attack_sds(dataset)
		detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]

		all_detection_points = pickle.load(open('ccs-storage/all-detection-points.pkl', 'rb'))
		model_all_detection_points = all_detection_points[lookup_name]
		
		all_mses = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
		val_mses = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')

		print(f'for {lookup_name}')
		print(f'avg_val_mse: {np.mean(val_mses)}')

		full_slice_values = np.zeros((len(sds), 150, ncols, 4))
		scatter_obj = np.zeros((len(sds), 16))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_type_list = []
		pattern_list = []
		atks_list = []
		detect_point_list = []
		length_list = []

		for sd_idx in range(len(sds)):
			sd_obj = sds[sd_idx]
			atk_idx = sd_obj[0]
			label = sd_obj[1]
			is_multi = sd_obj[3]
			sd = np.abs(sd_obj[4])
			col_idx = sensor_cols.index(label)

			att_start = np.min(attacks[atk_idx]) - history - 1
			att_end = np.max(attacks[atk_idx]) - history - 1

			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{atk_idx}-true150.pkl', 'rb')) 
			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)

			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{atk_idx}-true150.pkl', 'rb'))
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)

			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{atk_idx}-true150.pkl', 'rb')) 
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			# TODO: if needed, try a different slicing
			mse_rankings = np.zeros(150)
			sm_rankings = np.zeros(150)
			shap_rankings = np.zeros(150)
			lemna_rankings = np.zeros(150)
			
			slice_avg_rankings = np.zeros(150)
			slice_avg_values = np.zeros((150, 4))

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

				slice_avg = np.sum(np.vstack([mse_slice_norm, sm_slice_norm, lemna_slice_norm]), axis=0)
				slice_avg_rankings[i] = scores_to_rank(slice_avg, col_idx)

				slice_avg_values[i,0] = mse_slice_norm[col_idx]
				slice_avg_values[i,1] = sm_slice_norm[col_idx]
				slice_avg_values[i,2] = shap_slice_norm[col_idx]
				slice_avg_values[i,3] = lemna_slice_norm[col_idx]

				full_slice_values[sd_idx, i, :, 0] = mse_slice_norm
				full_slice_values[sd_idx, i, :, 1] = sm_slice_norm
				full_slice_values[sd_idx, i, :, 2] = shap_slice_norm
				full_slice_values[sd_idx, i, :, 3] = lemna_slice_norm

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

			# Record detection point, add to plot
			if atk_idx in model_detection_points:
				detect_point = model_detection_points[atk_idx]
				all_points = model_all_detection_points[atk_idx]
				
				# window = 5
				# binary_signal = np.zeros(np.max(all_points) + 1)
				# binary_signal[all_points] = 1
				# detection = np.convolve(binary_signal, np.ones(window), 'same') // window

				# if np.sum(detection) > 0:
				# 	new_detect_point = np.min(np.where(detection)) - 2

				# 	if detect_point != new_detect_point:
				# 		print(f'Attack {atk_idx} after windowing: detect point moved from {detect_point} to {new_detect_point}')
				# 	else:
				# 		print(f'Attack {atk_idx} same detect point {detect_point}.')

				# else:
				# 	print(f'Attack {atk_idx} is now missed.')

			else:

				detect_point = -1

			scatter_obj[sd_idx, 0] = sd
			scatter_obj[sd_idx, 1] = np.mean(mse_rankings)
			scatter_obj[sd_idx, 2] = np.min(mse_rankings)
			scatter_obj[sd_idx, 3] = mse_avg_ranking

			scatter_obj[sd_idx, 4] = np.mean(sm_rankings)
			scatter_obj[sd_idx, 5] = np.min(sm_rankings)
			scatter_obj[sd_idx, 6] = sm_avg_ranking

			scatter_obj[sd_idx, 7] = np.mean(shap_rankings)
			scatter_obj[sd_idx, 8] = np.min(shap_rankings)
			scatter_obj[sd_idx, 9] = shap_avg_ranking
			
			scatter_obj[sd_idx, 10] = np.mean(lemna_rankings)
			scatter_obj[sd_idx, 11] = np.min(lemna_rankings)
			scatter_obj[sd_idx, 12] = lemna_avg_ranking

			scatter_obj[sd_idx, 13] = np.mean(slice_avg_rankings)
			scatter_obj[sd_idx, 14] = np.min(slice_avg_rankings)
			scatter_obj[sd_idx, 15] = avg_avg_ranking

			if is_actuator(dataset, label):
				sensor_types_list.append('Actuator')
				val_type_list.append('bool')
			else:
				sensor_types_list.append('Sensor')
				val_type_list.append('float')

			detect_point_list.append(detect_point)
			labels_list.append(label)
			multi_list.append(is_multi)
			pattern_list.append(sd_obj[2])
			atks_list.append(atk_idx)
			length_list.append(len(attacks[atk_idx]))

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_type_list,
			'sd_type': pattern_list,
			'sd': scatter_obj[:,0],
			'attack_idx': atks_list,
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
		})

		pickle.dump(df, open(f'timing-{lookup_name}.pkl', 'wb'))
		pickle.dump(full_slice_values, open(f'full-values-{lookup_name}.pkl', 'wb'))

def parse_arguments():

	parser = utils.get_argparser()
	return parser.parse_args()

if __name__ == "__main__":

	make_ideal_plot_obj()
	make_detect_plot_obj()
	
	make_timing_plot_obj()
	make_detect_timing()

	print('Done')
