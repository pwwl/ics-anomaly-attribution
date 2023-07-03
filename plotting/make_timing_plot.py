
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib

import sys
sys.path.append('..')

import data_loader
from utils import tep_utils, attack_utils
from utils.attack_utils import get_attack_indices, get_attack_sds, is_actuator

matplotlib.rcParams['pdf.fonttype'] = 42
BETA_SCALE = 2.5

def slice_average(mse_scores, sm_scores, lemna_scores, dataset, sensor_cols):

	ncols = len(sensor_cols)
	slice_scores = np.zeros(ncols)

	for i in range(ncols):
		if is_actuator(dataset, sensor_cols[i]):
			slice_scores[i] = mse_scores[i] + BETA_SCALE * sm_scores[i] + BETA_SCALE * lemna_scores[i]
		else:
			slice_scores[i] = BETA_SCALE * mse_scores[i] + sm_scores[i] + lemna_scores[i]

	return slice_scores

def make_beta_plot(realdet=False, use_skips=False):

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI', 'TEP']
	betas = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
	ncols_arr = [51, 119, 53]

	for model in models:

		beta_plot_obj = []

		for ds in datasets:

			if model == 'CNN':
				lookup = f'CNN-{ds}-l2-hist50-kern3-units64-results_ns1'
			else:
				lookup = f'{model}-{ds}-l2-hist50-units64-results_ns1'

			print(f'For {model} {ds}')
			detection_points = pickle.load(open(f'meta-storage/{lookup}-detection-points.pkl', 'rb'))
			model_detection_points = detection_points[lookup]

			if realdet:
				full_slice = pickle.load(open(f'meta-storage/real-detection-timing-scores-{lookup}.pkl', 'rb'))
			else:
				full_slice = pickle.load(open(f'meta-storage/ideal-detection-timing-scores-{lookup}.pkl', 'rb'))
			
			if ds == 'TEP':
				sensor_cols = tep_utils.get_short_colnames()
				ncols = len(sensor_cols)
				attacks = tep_utils.get_footer_list(patterns=['cons'])
			else:
				Xtest, Ytest, sensor_cols = data_loader.load_test_data(ds)
				ncols = len(sensor_cols)
				attacks = get_attack_sds(ds)

			full_meth_wavg_ranking = np.zeros((150, len(betas)))

			num_counted = 0

			for att_idx in range(len(attacks)):
				
				if ds == 'TEP':
					atk_name = attacks[att_idx]
					splits = atk_name.split("_")
					label = splits[2]
				else:
					sd_obj = attacks[att_idx]
					atk_name = sd_obj[0]
					label = sd_obj[1]

				slice = full_slice[att_idx]
				col_idx = sensor_cols.index(label)

				if np.sum(slice) == 0:
					print(f'Attack {atk_name}: {label} missed.')
					continue
				
				if use_skips and model_detection_points[atk_name] < 1:
					print(f'Attack {atk_name}: {label} instant detect, skip.')
					continue

				print(f'For attack {atk_name}: {label}')
				
				# Ranking based on a weighted average of methods, taken every timestep
				ranking_wavg_of_methods = np.zeros((150, len(betas)))

				for bx in range(len(betas)):
					beta_val = betas[bx]
					slice_weighted_avg = np.zeros((150, ncols))
					
					for i in range(150):
						for k in range(ncols):
							if is_actuator(ds, sensor_cols[k]):
								slice_weighted_avg[i,k] = slice[i,k,0] + beta_val * slice[i,k,1] + beta_val * slice[i,k,2]
							else:
								slice_weighted_avg[i,k] = slice[i,k,0] + slice[i,k,1] + slice[i,k,2]

						ranking_wavg_of_methods[i,bx] = tep_utils.scores_to_rank(slice_weighted_avg[i], col_idx)

				full_meth_wavg_ranking += ranking_wavg_of_methods
				num_counted += 1

			full_meth_wavg_ranking /= num_counted
			print(f'Parsed {num_counted} attacks.')

			beta_plot_obj.append(np.mean(full_meth_wavg_ranking, axis=0))

		# Make beta plot
		beta_plot_obj = np.vstack(beta_plot_obj)

		##### Single plot
		fig, ax = plt.subplots(1, 1, figsize=(10, 6))
		
		for i in range(3):
			ax.plot(betas, beta_plot_obj[i] / ncols_arr[i], label=datasets[i], lw=3)
			ax.scatter(betas, beta_plot_obj[i] / ncols_arr[i])
			
		minyvals = np.min(beta_plot_obj, axis=1) / ncols_arr
		minxvals = np.argmin(beta_plot_obj, axis=1)

		for i in range(3):
			ax.scatter(betas[minxvals[i]], minyvals[i], color='red', marker='*', s=100, linewidths=5)
		
		ax.set_ylim([0, 0.35])
		ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
		ax.set_xticklabels([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
		ax.tick_params(axis='both', which='major', labelsize=24)

		ax.set_xlabel('Beta value', fontsize=32)
		ax.set_ylabel('Normalized AvgRank', fontsize=32)
		ax.legend(ncols=3, fontsize=28)

		if realdet and use_skips:
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-real-skips.pdf')
		elif realdet:
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-real.pdf')
		else:	
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-ideal.pdf')

		plt.close()

def make_timing_avg_plot(realdet=False, use_skips=False):

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI', 'TEP']
	ncols_arr = [51, 119, 53]

	for model in models:

		full_plot_obj = []
		full_avg_plot_obj = []
		full_wavg_plot_obj = []

		for ds in datasets:

			if model == 'CNN':
				lookup = f'CNN-{ds}-l2-hist50-kern3-units64-results_ns1'
			else:
				lookup = f'{model}-{ds}-l2-hist50-units64-results_ns1'

			print(f'For {model} {ds}')
			detection_points = pickle.load(open(f'meta-storage/{lookup}-detection-points.pkl', 'rb'))
			model_detection_points = detection_points[lookup]

			if realdet:
				full_slice = pickle.load(open(f'meta-storage/real-detection-timing-scores-{lookup}.pkl', 'rb'))
			else:
				full_slice = pickle.load(open(f'meta-storage/ideal-detection-timing-scores-{lookup}.pkl', 'rb'))
			
			if ds == 'TEP':
				sensor_cols = tep_utils.get_short_colnames()
				ncols = len(sensor_cols)
				attacks = tep_utils.get_footer_list(patterns=['cons'])
			else:
				Xtest, Ytest, sensor_cols = data_loader.load_test_data(ds)
				ncols = len(sensor_cols)
				attacks = get_attack_sds(ds)

			full_ranking_no_avg = np.zeros((150, 4))
			full_avg_score = np.zeros((150, 4))
			full_meth_avg_ranking = np.zeros(150)
			full_meth_wavg_ranking = np.zeros(150)
			num_counted = 0

			for att_idx in range(len(attacks)):
				
				if ds == 'TEP':
					atk_name = attacks[att_idx]
					splits = atk_name.split("_")
					label = splits[2]
				else:
					sd_obj = attacks[att_idx]
					atk_name = sd_obj[0]
					label = sd_obj[1]

				slice = full_slice[att_idx]
				col_idx = sensor_cols.index(label)

				if np.sum(slice) == 0:
					print(f'Attack {atk_name}: {label} missed.')
					continue

				if use_skips and model_detection_points[atk_name] < 1:
					print(f'Attack {atk_name}: {label} instant detect, skip.')
					continue

				# Ranking based on no averages
				rankings_no_avg = np.zeros((150, 4))
				
				# Ranking based on an average of methods, taken every timestep
				rankings_avg_of_methods = np.zeros(150)
				rankings_wavg_of_methods = np.zeros(150)
				
				#slice_avg = np.mean(slice, axis=2)
				slice_avg = np.mean(slice[:,:,[0,1,3]], axis=2)
				slice_weighted_avg = np.zeros((150, ncols))

				for i in range(150):

					for j in range(4):
						method_slice = slice[i, :, j]
						rankings_no_avg[i, j] = tep_utils.scores_to_rank(method_slice, col_idx)

					for k in range(ncols):
						if is_actuator(ds, sensor_cols[k]):
							slice_weighted_avg[i,k] = slice[i,k,0] + BETA_SCALE * slice[i,k,1] + BETA_SCALE * slice[i,k,3]
						else:
							slice_weighted_avg[i,k] = slice[i,k,0] + slice[i,k,1] + slice[i,k,3]

					rankings_wavg_of_methods[i] = tep_utils.scores_to_rank(slice_weighted_avg[i], col_idx)

					# Average of all three methods
					rankings_avg_of_methods[i] = tep_utils.scores_to_rank(slice_avg[i], col_idx)

				plot_attacked_feature_score = slice[:, col_idx, :]

				full_meth_avg_ranking += rankings_avg_of_methods
				full_meth_wavg_ranking += rankings_wavg_of_methods
				full_ranking_no_avg += rankings_no_avg
				full_avg_score += plot_attacked_feature_score
				num_counted += 1

			full_ranking_no_avg /= num_counted
			full_meth_avg_ranking /= num_counted
			full_meth_wavg_ranking /= num_counted
			full_avg_score /= num_counted
			print(f'Parsed {num_counted} attacks.')

			full_plot_obj.append(full_ranking_no_avg)
			full_avg_plot_obj.append(full_meth_avg_ranking)
			full_wavg_plot_obj.append(full_meth_wavg_ranking)

		full_plot_obj_norm = np.zeros((3, 150, 4))
		full_avg_plot_obj_norm = np.zeros((4, 150))
		full_wavg_plot_obj_norm = np.zeros((3, 150))

		### Convert to norm
		for i in range(3):
			full_plot_obj_norm[i] = full_plot_obj[i] / ncols_arr[i]
			full_avg_plot_obj_norm[i] = full_avg_plot_obj[i] / ncols_arr[i]
			full_wavg_plot_obj_norm[i] = full_wavg_plot_obj[i] / ncols_arr[i]

		fig, ax = plt.subplots(1, 1, figsize=(10, 6))
		ax.plot(np.arange(150), np.mean(full_plot_obj_norm, axis=0)[:,0], color='black', label='MSE', lw=2)
		ax.plot(np.arange(150), np.mean(full_plot_obj_norm, axis=0)[:,1], color='blue', label='SM', lw=2)
		ax.plot(np.arange(150), np.mean(full_plot_obj_norm, axis=0)[:,3], color='brown', label='LEMNA', lw=2)
		ax.plot(np.arange(150), np.mean(full_avg_plot_obj_norm, axis=0), color='red', label='Avg', lw=2)
		ax.legend(ncol=2, fontsize=28)
		ax.set_ylim([0, 0.5])
		ax.set_ylabel('Normalized AvgRank', fontsize=32)
		ax.tick_params(axis='both', which='major', labelsize=24)

		if realdet and use_skips:
			ax.set_xlabel('Time from detection point (seconds)', fontsize=32)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-real-skips.pdf')
		elif realdet:
			ax.set_xlabel('Time from detection point (seconds)', fontsize=32)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-real.pdf')
		else:
			ax.set_xlabel('Time from attack start (seconds)', fontsize=32)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-ideal.pdf')

		plt.close()

if __name__ == '__main__':
	
	make_timing_avg_plot()
	make_beta_plot()
	
	make_timing_avg_plot(realdet=True)
	make_timing_avg_plot(realdet=True, use_skips=True)
	make_beta_plot(realdet=True)
	make_beta_plot(realdet=True, use_skips=True)
