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

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import sys
sys.path.append('explain-eval-manipulations')

from tep_utils import scores_to_rank
import tep_utils

np.set_printoptions(suppress=True)
DEFAULT_CMAP = plt.get_cmap('Reds', 5)

HOUR = 2000
SCALE = 1

def make_detect_plot_obj_tep(lookup_tupls):

	attacks = tep_utils.get_footer_list(patterns=['cons'])

	detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))

	for lookup_name, history in lookup_tupls:

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

			all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			detect_idx = model_detection_points[attack_footer]
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 

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

		pickle.dump(df, open(f'realdet-{lookup_name}.pkl', 'wb'))

def make_tep_ideal_plot_obj(lookup_tupls):

	attack_footers = tep_utils.get_footer_list(patterns=['cons'])
	sensor_cols = tep_utils.get_short_colnames()

	for lookup_name, history in lookup_tupls:

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
			all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{attack_footer}-ns.npy')

			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			
			smap_scores = np.sum(np.abs(smap_scores_full), axis=1)
			shap_scores = np.sum(np.abs(shap_scores_full), axis=1)
			lemna_scores = np.sum(np.abs(lemna_scores_full), axis=1)

			# Ignoring detections
			first_mses = all_mses[att_start+history]
			first_sm = smap_scores[0]
			first_shap = shap_scores[0]
			first_lemna = lemna_scores[0]
			total_lemna += np.mean(first_lemna)

			first_ranking = scores_to_rank(first_mses, col_idx)
			first_sm_ranking = scores_to_rank(first_sm, col_idx)
			first_shap_ranking = scores_to_rank(first_shap, col_idx)
			first_lemna_ranking = scores_to_rank(first_lemna, col_idx)

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
		print('------------------------')

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'val_type': val_type_list,
			'sd_type': pattern_list,
			'is_multi': multi_list,
			'mse_ranking': first_scatter_obj[:,0],
			'smap_ranking': first_scatter_obj[:,1],
			'shap_ranking': first_scatter_obj[:,2],
			'lemna_ranking': first_scatter_obj[:,3],
		})

		sen_idx = np.where(df['sensor_type'] == 'Sensor')[0]

		pickle.dump(df, open(f'idealdet-{lookup_name}.pkl', 'wb'))

def make_tep_timing_plot_obj(lookup_tupls):
	
	for lookup_name, history in lookup_tupls:

		detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]

		all_detection_points = pickle.load(open('meta-storage/all-detection-points.pkl', 'rb'))
		model_all_detection_points = all_detection_points[lookup_name]

		attack_footers = tep_utils.get_footer_list(patterns=['cons'])

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

			all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-true150.pkl', 'rb')) 

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

		pickle.dump(df, open(f'timing-{lookup_name}.pkl', 'wb'))
		pickle.dump(full_slice_values, open(f'full-values-{lookup_name}.pkl', 'wb'))
	

def make_detect_timing(lookup_tupls):

	attacks = tep_utils.get_footer_list(patterns=['cons'])
	sensor_cols = tep_utils.get_short_colnames()
	ncols = len(sensor_cols)

	detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))

	for lookup_name, history in lookup_tupls:
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

			all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
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

		pickle.dump(df, open(f'real-timing-{lookup_name}.pkl', 'wb'))
		pickle.dump(full_slice_values, open(f'full-values-real-{lookup_name}.pkl', 'wb'))

def make_stealth_plot_obj(lookup_tupls):

	for lookup_name, history in lookup_tupls:

		attacks = tep_utils.get_footer_list(patterns=['cons', 'csum', 'line'], mags=['p2s'], locations='pid')

		detection_points = pickle.load(open('meta-storage/detection-points.pkl', 'rb'))
		model_detection_points = detection_points[lookup_name]
		
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

			all_mses = np.load(f'meta-storage/model-mses/mses-{lookup_name}-{attack_footer}-ns.npy')
			detect_idx = model_detection_points[attack_footer]
			smap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-saliency_map_mse_history-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			shap_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-SHAP-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 
			lemna_scores_full = pickle.load(open(f'explanations-dir/explain23-detect-pkl/explanations-LEMNA-{lookup_name}-{attack_footer}-detect150.pkl', 'rb')) 

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

			detect_list.append(detect_idx)
			labels_list.append(label)
			multi_list.append(is_multi)
			pattern_list.append(pattern)

		det_idx = first_scatter_obj[:,0] > 0

		df = pd.DataFrame({
			'sensor': labels_list,
			'sensor_type': sensor_types_list,
			'sd_type': pattern_list,
			'is_multi': multi_list,
			'detect_point': detect_list,
			'mse_ranking': first_scatter_obj[det_idx,0],
			'smap_ranking': first_scatter_obj[det_idx,1],
			'shap_ranking': first_scatter_obj[det_idx,2],
			'lemna_ranking': first_scatter_obj[det_idx,3],
		})

		pickle.dump(df, open(f'realdet-stealth-{lookup_name}.pkl', 'wb'))

def parse_arguments():
	
	parser = argparse.ArgumentParser()
	model_choices = set(['CNN', 'GRU', 'LSTM'])
	
	parser.add_argument("--md", 
		help="Format as {model}-{dataset}-l{layers}-hist{history}-kern{kernel}-units{units}-{runname} if model is CNN, " +
			 "format as {model}-{dataset}-l{layers}-hist{history}-units{units}-{runname} otherwise. " +
			 "As an example: CNN-TEP-l2-hist50-kern3-units64-results",
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
		if dataset != 'TEP':
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

	return lookup_names

if __name__ == "__main__":

	lookup_tupls = parse_arguments()

	make_tep_ideal_plot_obj(lookup_tupls)
	make_detect_plot_obj_tep(lookup_tupls)

	make_tep_timing_plot_obj(lookup_tupls)
	make_detect_timing(lookup_tupls)

	make_stealth_plot_obj(lookup_tupls)

	print('Done!')
