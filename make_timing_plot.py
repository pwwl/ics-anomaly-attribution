
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib

import data_loader
import tep_utils
from attack_utils import get_attack_indices, get_attack_sds, get_rel_scores, is_actuator

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

def averaging_dfs():

	datasets = ['SWAT', 'WADI', 'TEP']
	models = ['CNN', 'GRU', 'LSTM']
	
	idealtime_df = []
	ideal_df = []

	realtime_df = []
	real_df = []

	slice_obj = []

	for model in models:
		
		for ds in datasets:
			
			if model == 'CNN':
				lookup = f'CNN-{ds}-l2-hist50-kern3-units64-results_ns1'
			else:
				lookup = f'{model}-{ds}-l2-hist50-units64-results_ns1'

			dfi = pickle.load(open(f'meta-storage/model-detection-ranks/idealdet-{lookup}.pkl', 'rb'))
			df_it = pickle.load(open(f'timing-{lookup}.pkl', 'rb'))
			df_it['best_ranking'] = np.min(df_it[['mse_best_ranking', 'sm_best_ranking', 'shap_best_ranking', 'lemna_best_ranking']].values, axis=1)
			df_it['dataset'] = ds

			idealtime_df.append(df_it)
			ideal_df.append(dfi)

			# dfr = pickle.load(open(f'realdet-{lookup}.pkl', 'rb'))
			# df_rt = pickle.load(open(f'real-timing-{lookup}.pkl', 'rb'))
			# df_rt['best_ranking'] = np.min(df_rt[['mse_best_ranking', 'sm_best_ranking', 'lemna_best_ranking']].values, axis=1)
			# df_rt['dataset'] = ds

			# realtime_df.append(df_rt)
			# real_df.append(dfr)

		df1 = pd.concat(idealtime_df)
		df2 = pd.concat(ideal_df)
		# df3 = pd.concat(realtime_df)
		# df4 = pd.concat(real_df)

		print(f'---- IDEAL {model} {ds} -----')
		print(f'Full averaging ranking: {np.mean(df1["slice_tavg_ranking"])}')
		print(f'MSE averaging ranking: {np.mean(df1["mse_tavg_ranking"])}')
		print(f'SM averaging ranking: {np.mean(df1["sm_tavg_ranking"])}')
		print(f'SHAP averaging ranking: {np.mean(df1["shap_tavg_ranking"])}')
		print(f'LEMNA averaging ranking: {np.mean(df1["lemna_tavg_ranking"])}')
		
		print(f'MSE ranking: {np.mean(df2["mse_ranking"])}')
		print(f'SM ranking: {np.mean(df2["sm_ranking"])}')
		print(f'SHAP ranking: {np.mean(df2["shap_ranking"])}')
		print(f'LEMNA ranking: {np.mean(df2["lemna_ranking"])}')

		# print('---- DETECTED -----')
		# print(f'Full averaging ranking: {np.mean(df3["slice_tavg_ranking"])}')
		# print(f'MSE averaging ranking: {np.mean(df3["mse_tavg_ranking"])}')
		# print(f'SM averaging ranking: {np.mean(df3["sm_tavg_ranking"])}')
		# print(f'LEMNA averaging ranking: {np.mean(df3["lemna_tavg_ranking"])}')
		# print(f'MSE ranking: {np.mean(df4["mse_ranking"])}')
		# print(f'SM ranking: {np.mean(df4["sm_ranking"])}')
		# print(f'LEMNA ranking: {np.mean(df4["lemna_ranking"])}')
		
		# for ds in datasets:
		# 	print(f"{ds} detected: {np.sum(df3['dataset'] == ds)}")
		# 	print(f"{ds} instant detected: {np.sum((df3['detect_point'] < 5) & (df3['dataset'] == ds))}")

	return df1, df2

def make_beta_plot(realdet=False, use_skips=False):

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI', 'TEP']
	betas = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
	ncols_arr = [51, 119, 53]
	detection_points = pickle.load(open(f'meta-storage/detection-points.pkl', 'rb'))

	for model in models:

		beta_plot_obj = []

		for ds in datasets:

			if model == 'CNN':
				lookup = f'CNN-{ds}-l2-hist50-kern3-units64-results_ns1'
			else:
				lookup = f'{model}-{ds}-l2-hist50-units64-results_ns1'

			print(f'For {model} {ds}')
			model_detection_points = detection_points[lookup]

			if realdet:
				full_slice = pickle.load(open(f'meta-storage/full-values-real-{lookup}.pkl', 'rb'))
			else:
				full_slice = pickle.load(open(f'meta-storage/full-values-{lookup}.pkl', 'rb'))
			
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
								#slice_weighted_avg[i,k] = beta_val * slice[i,k,0] + (1 / beta_val) * slice[i,k,1] + (1 / beta_val) * slice[i,k,2]
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
		#ax.legend(ncols=3, fontsize=24, loc='lower center', bbox_to_anchor=(0.5, 1))
		ax.legend(ncols=3, fontsize=28)

		if realdet and use_skips:
			#ax.set_title('Beta-weighted avgRank - from first detection point', fontsize=18)
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-realdet-skips.pdf')
		elif realdet:
			#ax.set_title('Beta-weighted avgRank - from first detection point', fontsize=18)
			#fig.tight_layout(rect=[0, 0, 1, 1])
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-realdet.pdf')
		else:	
			#ax.set_title('Beta-weighted avgRank - from attack start', fontsize=18)
			#fig.tight_layout(rect=[0, 0, 1, 1])
			fig.tight_layout()
			plt.savefig(f'plot-beta-{model}-idealdet.pdf')

		plt.close()

def make_timing_avg_plot(realdet=False, use_skips=False):

	models = ['CNN', 'GRU', 'LSTM']
	datasets = ['SWAT', 'WADI', 'TEP']
	ncols_arr = [51, 119, 53]
	#dfi, dfr = averaging_dfs()
	detection_points = pickle.load(open(f'meta-storage/detection-points.pkl', 'rb'))

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
			model_detection_points = detection_points[lookup]

			if realdet:
				full_slice = pickle.load(open(f'meta-storage/full-values-real-{lookup}.pkl', 'rb'))
			else:
				full_slice = pickle.load(open(f'meta-storage/full-values-{lookup}.pkl', 'rb'))
			
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

				#print(f'For attack {atk_name}: {label}')

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
							#slice_weighted_avg[i,k] = BETA_SCALE * slice[i,k,0] + slice[i,k,1] + slice[i,k,2]
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

				# fig, ax = plt.subplots(2, 1, figsize=(10, 12))
				# ax[0].plot(np.arange(150), rankings_no_avg[:,0], color='black', label='MSE')
				# ax[0].plot(np.arange(150), rankings_no_avg[:,1], color='blue', label='SM')
				# ax[0].plot(np.arange(150), rankings_no_avg[:,2], color='brown', label='LEMNA')
				# ax[0].plot(np.arange(150), rankings_avg_of_methods, color='red', label='Avg')
				# ax[0].plot(np.arange(150), rankings_wavg_of_methods, color='orange', label=f'Avg (B = {BETA_SCALE})')
				# ax[0].legend()
				# ax[0].set_ylim([0, ncols])
				# ax[0].set_ylabel('Attack feature ranking', fontsize=16)
				
				# ax[1].plot(np.arange(150), plot_attacked_feature_score[:,0], color='black', label='MSE')
				# ax[1].plot(np.arange(150), plot_attacked_feature_score[:,1], color='blue', label='SM')
				# ax[1].plot(np.arange(150), plot_attacked_feature_score[:,2], color='brown', label='LEMNA')

				# ax[1].legend()
				# ax[1].set_ylim([0, 1])
				# ax[1].set_ylabel('Attack feature \n anomaly score', fontsize=16)
				# fig.tight_layout()
				# plt.savefig(f'temp-timing/timing-{ds}-{atk_name}-{label}.png')
				# plt.close()

			full_ranking_no_avg /= num_counted
			full_meth_avg_ranking /= num_counted
			full_meth_wavg_ranking /= num_counted
			full_avg_score /= num_counted
			print(f'Parsed {num_counted} attacks.')

			###### Averaging by time

			# fig, ax = plt.subplots(2, 1, figsize=(10, 12))
			# ax[0].plot(np.arange(150), full_ranking_no_avg[:,0], color='black', label='MSE', lw=2)
			# ax[0].plot(np.arange(150), full_ranking_no_avg[:,1], color='blue', label='SM', lw=2)
			# ax[0].plot(np.arange(150), full_ranking_no_avg[:,2], color='brown', label='LEMNA', lw=2)
			# ax[0].plot(np.arange(150), full_meth_avg_ranking, color='red', label='Avg', lw=2)
			# ax[0].plot(np.arange(150), full_meth_wavg_ranking, color='orange', label=f'Avg (B = {BETA_SCALE})', lw=2)
			# ax[0].legend(fontsize=14)
			# ax[0].set_ylim([0, ncols])
			# ax[0].set_ylabel('Time-averaged ranking', fontsize=16)
			
			# ax[1].plot(np.arange(150), full_avg_score[:,0], color='black', label='MSE', lw=2)
			# ax[1].plot(np.arange(150), full_avg_score[:,1], color='blue', label='SM', lw=2)
			# ax[1].plot(np.arange(150), full_avg_score[:,2], color='brown', label='LEMNA', lw=2)

			# ax[1].legend(fontsize=14)
			# ax[1].set_ylim([0, 1])
			# ax[1].set_ylabel('Time-averaged anomaly score', fontsize=16)

			# if realdet and use_skips:
			# 	ax[0].set_title(f'{ds} AvgRank - Filtered', fontsize=18)
			# 	ax[0].set_xlabel('Time from first detection point (seconds)', fontsize=16)
			# 	fig.tight_layout()
			# 	plt.savefig(f'plot-timing-{model}-{ds}-realdet-skips.pdf')
			# elif realdet:
			# 	ax[0].set_title(f'{ds} AvgRank', fontsize=18)
			# 	ax[0].set_xlabel('Time from first detection point (seconds)', fontsize=16)
			# 	fig.tight_layout()
			# 	plt.savefig(f'plot-timing-{model}-{ds}-realdet.pdf')
			# else:
			# 	ax[0].vlines(50, ymin=0, ymax=1, color='grey', linestyle='--')
			# 	ax[0].text(52, 0.85, 'timestep when\nΔt = history', fontsize=14)
			# 	ax[0].set_title(f'{ds} AvgRank', fontsize=18)
			# 	ax[0].set_xlabel('Time from attack start (seconds)', fontsize=16)
			# 	fig.tight_layout()
			# 	plt.savefig(f'plot-timing-{model}-{ds}-idealdet.pdf')

			# plt.close()

			#print(np.sum(full_ranking_no_avg))

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
		#ax.plot(np.arange(150), np.mean(full_plot_obj_norm, axis=0)[:,2], color='green', label='SHAP', lw=2)
		ax.plot(np.arange(150), np.mean(full_plot_obj_norm, axis=0)[:,3], color='brown', label='LEMNA', lw=2)
		ax.plot(np.arange(150), np.mean(full_avg_plot_obj_norm, axis=0), color='red', label='Avg', lw=2)
		#ax.plot(np.arange(150), np.mean(full_wavg_plot_obj_norm, axis=0), color='orange', label=f'Avg (B = {BETA_SCALE})', lw=2)
		ax.legend(ncol=2, fontsize=28)
		ax.set_ylim([0, 0.5])
		ax.set_ylabel('Normalized AvgRank', fontsize=32)
		ax.tick_params(axis='both', which='major', labelsize=24)

		if realdet and use_skips:
			#ax.set_title(f'{model} AvgRank - Filtered', fontsize=24)
			ax.set_xlabel('Time from detection point (seconds)', fontsize=32)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-realdet-skips.pdf')
		elif realdet:
			#ax.set_title(f'{model} practical timing', fontsize=32)
			ax.set_xlabel('Time from detection point (seconds)', fontsize=32)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-realdet.pdf')
		else:
			#ax.set_title(f'{model} ideal timing', fontsize=32)
			ax.set_xlabel('Time from attack start (seconds)', fontsize=32)
			#ax.vlines(50, ymin=0, ymax=1, color='grey', linestyle='--')
			#ax.text(52, 0.4, 'model input\nstarts at anomaly', fontsize=18)
			fig.tight_layout()
			plt.savefig(f'plot-timing-{model}-idealdet.pdf')

		plt.close()

if __name__ == '__main__':
	
	make_timing_avg_plot()
	#make_beta_plot()
	
	make_timing_avg_plot(realdet=True)
	#make_timing_avg_plot(realdet=True, use_skips=True)
	#make_beta_plot(realdet=True)
	#make_beta_plot(realdet=True, use_skips=True)

	# averaging_dfs()
