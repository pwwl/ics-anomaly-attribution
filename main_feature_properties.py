import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pickle

import sys
sys.path.append('explain-eval-attacks')

from data_loader import load_train_data, load_test_data
from main_train import load_saved_model

from attack_utils import get_attack_indices, get_attack_sds, get_rel_scores, is_actuator
from tep_utils import scores_to_rank

import data_loader
import argparse

np.set_printoptions(suppress=True)
DEFAULT_CMAP = plt.get_cmap('Reds', 5)

HOUR = 2000
SCALE = 1

def make_detect_plot_obj(lookup_tupls, attacks_to_consider):

	all_dfs = []
	all_plots = []

	for lookup_name, dataset, history in lookup_tupls:

		Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)

		print(f'Processing {dataset} dataset')
		
		attacks, labels = get_attack_indices(dataset)
		detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]
		sds = get_attack_sds(dataset)
		sds_to_consider = []
		for sd in sds:
			if sd[0] in attacks_to_consider:
				sds_to_consider.append(sd)
		
		all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-ns.npy')
		val_mses = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-ns.npy')

		print(f'for {lookup_name}')
		print(f'avg_val_mse: {np.mean(val_mses)}')

		first_scatter_obj = np.zeros((len(sds_to_consider), 4))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_types_list = []
		pattern_list = []
		detect_list = []

		for sd_idx in range(len(sds_to_consider)):
			
			sd_obj = sds_to_consider[sd_idx]
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

			first_scatter_obj[sd_idx, 0] = first_ranking
			first_scatter_obj[sd_idx, 1] = first_sm_ranking
			first_scatter_obj[sd_idx, 2] = first_shap_ranking
			first_scatter_obj[sd_idx, 3] = first_lemna_ranking

			print(f'Attack {sd_obj}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

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
			'is_multi': multi_list,
			'detect_point': detect_list,
			'mse_ranking': first_scatter_obj[det_idx,0],
			'sm_ranking': first_scatter_obj[det_idx,1],
			'shap_ranking': first_scatter_obj[det_idx,2],
			'lemna_ranking': first_scatter_obj[det_idx,3]
		})

		pickle.dump(df, open(f'meta-storage/realdet-{lookup_name}.pkl', 'wb'))

def make_detect_timing(lookup_tupls, attacks_to_consider):

	all_dfs = []
	all_plots = []
	
	for lookup_name, dataset, history in lookup_tupls:

		Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)
		ncols = len(sensor_cols)

		print(f'Processing {dataset} dataset')
		
		attacks, labels = get_attack_indices(dataset)
		detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]
		sds = get_attack_sds(dataset)
		sds_to_consider = []
		for sd in sds:
			if sd[0] in attacks_to_consider:
				sds_to_consider.append(sd)
		
		all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-ns.npy')
		val_mses = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-ns.npy')

		print(f'for {lookup_name}')
		print(f'avg_val_mse: {np.mean(val_mses)}')

		scatter_obj = np.zeros((len(sds_to_consider), 16))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_types_list = []
		pattern_list = []
		detect_list = []
		length_list = []
		full_slice_values = np.zeros((len(sds_to_consider), 150, ncols, 4))

		for sd_idx in range(len(sds_to_consider)):
			
			sd_obj = sds_to_consider[sd_idx]
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

		pickle.dump(df, open(f'meta-storage/real-timing-{lookup_name}.pkl', 'wb'))
		pickle.dump(full_slice_values, open(f'meta-storage/full-values-real-{lookup_name}.pkl', 'wb'))

def make_ideal_plot_obj(lookup_tupls, attacks_to_consider):

	all_dfs = []
	all_plots = []

	for lookup_name, dataset, history in lookup_tupls:

		Xtest, Ytest, sensor_cols = data_loader.load_test_data(dataset)
		print(f'Processing {dataset} dataset')

		attacks, labels = get_attack_indices(dataset)
		sds = get_attack_sds(dataset)
		sds_to_consider = []
		for sd in sds:
			if sd[0] in attacks_to_consider:
				sds_to_consider.append(sd)
		
		all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-ns.npy')
		val_mses = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-ns.npy')

		print(f'for {lookup_name}')
		print(f'avg_val_mse: {np.mean(val_mses)}')

		first_scatter_obj = np.zeros((len(sds_to_consider), 4))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_type_list = []
		pattern_list = []

		for sd_idx in range(len(sds_to_consider)):
			sd_obj = sds_to_consider[sd_idx]
			atk_idx = sd_obj[0]
			label = sd_obj[1]
			is_multi = sd_obj[3]
			sd = np.abs(sd_obj[4])
			col_idx = sensor_cols.index(label)

			att_start = np.min(attacks[atk_idx][0]) - history - 1
			att_end = np.max(attacks[atk_idx]) - history - 1

			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{atk_idx}-true5.pkl', 'rb')) # CHANGE THIS TO true5
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{atk_idx}-true5.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{atk_idx}-true5.pkl', 'rb')) 

			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			# Ignoring detections
			first_mses = all_mses[att_start+history]
			first_sm = smap_scores[0]											# 51 OR 0? PROBABLY 0
			first_shap = shap_scores[0]
			first_lemna = lemna_scores[0]

			first_ranking = scores_to_rank(first_mses, col_idx)
			first_sm_ranking = scores_to_rank(first_sm, col_idx)
			first_shap_ranking = scores_to_rank(first_shap, col_idx)
			first_lemna_ranking = scores_to_rank(first_lemna, col_idx)

			print(f'Attack {sd_obj}: MSE-Rank {first_ranking}, SM-Rank {first_sm_ranking}, SHAP-Rank {first_shap_ranking}, LEMNA-Rank {first_lemna_ranking}')

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

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_type_list,
			'sd_type': pattern_list,	
			'is_multi': multi_list,
			'mse_ranking': first_scatter_obj[:,0],
			'sm_ranking': first_scatter_obj[:,1],
			'shap_ranking': first_scatter_obj[:,2],
			'lemna_ranking': first_scatter_obj[:,3],
		})

		pickle.dump(df, open(f'meta-storage/model-detection-ranks/idealdet-{lookup_name}.pkl', 'wb'))

def make_timing_plot_obj(lookup_tupls, attacks_to_consider):
		
	for lookup_name, dataset, history in lookup_tupls:

		_, _, sensor_cols = data_loader.load_test_data(dataset)
		ncols = len(sensor_cols)

		print(f'Processing {dataset} dataset')
		attacks, labels = get_attack_indices(dataset)
		sds = get_attack_sds(dataset)
		sds_to_consider = []
		for sd in sds:
			if sd[0] in attacks_to_consider:
				sds_to_consider.append(sd)
		detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]

		all_detection_points = pickle.load(open('meta-storage/all-detection-points.pkl', 'rb'))
		model_all_detection_points = all_detection_points[lookup_name]
		
		all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-ns.npy')
		val_mses = np.load(f'meta-storage/model-mses/mses-val-{lookup_name}-ns.npy')

		print(f'for {lookup_name}')
		print(f'avg_val_mse: {np.mean(val_mses)}')

		full_slice_values = np.zeros((len(sds_to_consider), 150, ncols, 4))
		scatter_obj = np.zeros((len(sds_to_consider), 16))
		
		labels_list = []
		multi_list = []
		sensor_types_list = []
		val_type_list = []
		pattern_list = []
		atks_list = []
		detect_point_list = []
		length_list = []

		for sd_idx in range(len(sds_to_consider)):
			sd_obj = sds_to_consider[sd_idx]
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
		pickle.dump(full_slice_values, open(f'meta-storage/full-values-{lookup_name}.pkl', 'wb'))

def parse_arguments():
	
	parser = argparse.ArgumentParser()
	model_choices = set(['CNN', 'GRU', 'LSTM'])
	data_choices = set(['SWAT', 'WADI'])

	parser.add_argument("attack",
		help="Which attack to explore?",
		type=int,
		nargs='+')
	
	parser.add_argument("--md", 
		help="Format as {model}-{dataset}-l{layers}-hist{history}-kern{kernel}-units{units}-{runname} if model is CNN, " +
			 "format as {model}-{dataset}-l{layers}-hist{history}-units{units}-{runname} otherwise. " +
			 "As an example: CNN-SWAT-l2-hist50-kern3-units64-results",
		nargs='+')

	lookup_names = []
	for arg in parser.parse_args().md:
		vals = arg.split("-")
		numArgs = len(vals)
		
		# if incorrect number of arguments
		if numArgs != 6 and numArgs != 7:
			raise SystemExit(f"ERROR: Provided incorrectly formatted argument {arg}")
		
		model_type, dataset, layers, history = vals[:4]
		units = vals[-2]

		if model_type not in model_choices:
			raise SystemExit(f"ERROR: Provided invalid model type {model_type}")
		if dataset not in dataset_choices:
			raise SystemExit(f"ERROR: Provided invalid dataset name {dataset}")
		if not units.startswith("units") or not units[len("units"):].isnumeric():
			raise SystemExit(f"ERROR: Provided invalid # of units in hidden layers {units}")
		if not history.startswith("hist") or not history[len("hist"):].isnumeric():
			raise SystemExit(f"ERROR: Provided invalid history length {history}")
		if not layers.startswith("l") or not layers[len("l"):].isnumeric():
			raise SystemExit(f"ERROR: Provided invalid # of layers {layers}")
		run_name = vals[-1]
		# if model is CNN (has kernel argument)
		if numArgs == 7:
			kernel = vals[4]
			if not kernel.startswith("kern") or not kernel[len("kern"):].isnumeric():
				raise SystemExit(f"ERROR: Provided invalid kernel size {kernel}")
		
		lookup_names.append((arg, int(history[len("hist"):])))

	return lookup_names, set(parser.parse_args().attack)

if __name__ == "__main__":


	lookup_tupls, attacks = parse_arguments()

	make_ideal_plot_obj(lookup_tupls, attacks)
	make_detect_plot_obj(lookup_tupls, attacks)
	
	make_timing_plot_obj(lookup_tupls, attacks)
	make_detect_timing(lookup_tupls, attacks)

	print('Done')
