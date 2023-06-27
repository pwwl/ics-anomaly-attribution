import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pdb
import pickle

# Internal imports
import os
import sys
sys.path.append('..')

import tep_utils
import tep_plot_utils

from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, scores_to_rank
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list

from data_loader import load_train_data, load_test_data
import attack_utils

# Clean up wd, set options
np.set_printoptions(suppress=True)
plt.style.use('ggplot')
DEFAULT_CMAP = plt.get_cmap('Reds', 5)
att_skip_list = get_skip_list()

#QUAL_CMAP = ['#003f5c','#2f4b7c','#665191','#a05195','#d45087','#f95d6a','#ff7c43','#ffa600']

normal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
for i in range(12):
	(c1, c2, c3) = normal12[i]
	normal12[i] = (c1 / 255, c2 / 255, c3 / 255)

#os.chdir('..')
_, sensor_cols_swat = load_train_data('SWAT')
_, sensor_cols_swatc = load_train_data('SWAT-CLEAN')
_, sensor_cols_swatp = load_train_data('SWAT-PHY')
_, sensor_cols_tepk = load_train_data('TEPK')
_, sensor_cols_wadi = load_train_data('WADI')
_, sensor_cols_wadip = load_train_data('WADI-PHY')

sensor_cols = dict()
sensor_cols['SWAT'] = sensor_cols_swat
sensor_cols['SWAT-CLEAN'] = sensor_cols_swatc
sensor_cols['SWAT-PHY'] = sensor_cols_swatp
sensor_cols['TEPK'] = sensor_cols_tepk
sensor_cols['WADI'] = sensor_cols_wadi
sensor_cols['WADI-PHY'] = sensor_cols_wadip

def idx_to_sen(idx, dataset):
	return sensor_cols[dataset][idx]

def sen_to_idx(sensor, dataset):

	for i in range(len(sensor_cols[dataset])):
		if sensor_cols[dataset][i] == sensor:
			return i

	return -1

def full_neighbor_score_sweep(graph, dataset, vector_scores, correlation=False, n_iterations=1):

	neighbor_scores = np.copy(vector_scores)

	for i in range(n_iterations):

		for idx in range(len(neighbor_scores)):

			sensor = idx_to_sen(idx, dataset)

			# Find the total covariance (for weighted sum)
			total_corr = 1
			for child in graph[sensor].keys():
				total_corr += graph[sensor][child]['weight']

			if correlation:
				total_score = neighbor_scores[idx] * (1 / total_corr)
			else:
				total_score = neighbor_scores[idx]

			# Take sum of all children
			for child in graph[sensor].keys():

				quant_score = neighbor_scores[sen_to_idx(child, dataset)]
				corr_score = graph[sensor][child]['weight']

				if correlation:
					total_score += quant_score * (corr_score / total_corr)
				else:
					total_score += quant_score

			neighbor_scores[idx] = total_score / (len(graph[sensor].keys()) + 1)

	return neighbor_scores

# TODO Make sen to idx compatible with all datasets
def score_attack_explanation(graph, dataset, candidates, vector_scores, attack_target, verbose=0):

	# 3 for exact MSE, 2 for Max graph chain, 1 for graph neighbor chain
	scoring_outcome = 0

	if len(candidates) == 0:
		return -1, 0, 0

	else:

		# Find all the chains, find out which parent is the max.
		max_parent_score = 0
		max_parent_cand = ()
		graph_chains = dict()

		ns = full_neighbor_score_sweep(graph, dataset, vector_scores)
		vs = vector_scores

	if sen_to_idx(attack_target, dataset) == -1:
		pdb.set_trace()

	vector_rank = scores_to_rank(vs, sen_to_idx(attack_target, dataset))
	neighbor_rank = scores_to_rank(ns, sen_to_idx(attack_target, dataset))

	return scoring_outcome, vector_rank, neighbor_rank

def open_and_explore_graphrank(filename, dataset_name, graph=None, suffix='in', graph_suffix='filtered', verbose=0, selection_idx=0, use_averaging=False):

	if graph is None:
		graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-{dataset_name}-{graph_suffix}.gml')

	graphrank_results = pickle.load(open(f'explanations-dir/{filename}', 'rb'))

	_, labels = attack_utils.get_attack_indices(dataset_name)

	if dataset_name == 'TEPK':
		labels = labels[0:5]

	per_attack_scoring = []
	per_attack_vr = []
	per_attack_nr = []
	
	for idx in range(len(labels)):

		labels_list = labels[idx]

		for label in labels_list:

			if verbose >= 0:
				print('==================================')
				print(f'Scoring: attack {idx} {label}')
				print('==================================')
			
			attack_code = f'{idx}_{suffix}'
			attack_quant = f'{idx}_quant_{suffix}'

			if len(graphrank_results[attack_code]) > 0:

				if use_averaging:
					selected_quant = np.abs(graphrank_results[attack_quant][:, 0:selection_idx])
					this_graph_quant = np.mean(selected_quant, axis=1)
				else:
					this_graph_quant = np.abs(graphrank_results[attack_quant][:, selection_idx])

				outcome, vector_rank, neighbor_rank = score_attack_explanation(graph, dataset_name, graphrank_results[attack_code], this_graph_quant, label, verbose=verbose)

				# print(f'Scoring outcome: {outcome}')
				# print(f'Scoring outcome rank: {vector_rank}')

				per_attack_scoring.append(outcome)
				per_attack_vr.append(vector_rank)
				per_attack_nr.append(neighbor_rank)

			else:

				per_attack_scoring.append(-1)
				per_attack_vr.append(0)
				per_attack_nr.append(0)

				# per_attack_scoring[idx] = -1
				# per_attack_vr[idx] = 0
				# per_attack_nr[idx] = 0

	return np.array(per_attack_scoring), np.array(per_attack_vr), np.array(per_attack_nr)

def plot_original_grouped_cdf(input_config, with_graphs=True, plot=False):

	cdf_objs = []
	# _, _, sensor_cols = load_test_data('SWAT')
	n_cols = len(sensor_cols['SWAT-CLEAN'])
	graph_suffix = 'filtered'

	for config in input_config:

		dict_header = config["swat-filename"][:-4]
		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['swat-filename'],
				'SWAT-CLEAN',
				suffix=config['suffix'],
				graph_suffix=graph_suffix,
				verbose=-1)

		_, per_attack_vr2, per_attack_nr2 = open_and_explore_graphrank(config['tepk-filename'],
				'TEPK',
				suffix=config['suffix'],
				graph_suffix=graph_suffix,
				verbose=-1)

		print(np.sum(per_attack_vr == 1) + np.sum(per_attack_vr2 == 1))

		# Normalize rankings such that they are same length
		per_attack_vr2_norm = (np.array(per_attack_vr2) / 53) * n_cols
		per_attack_nr2_norm = (np.array(per_attack_nr2) / 53) * n_cols

		full_per_attack_vr = np.concatenate([per_attack_vr, per_attack_vr2_norm])
		full_per_attack_nr = np.concatenate([per_attack_nr, per_attack_nr2_norm])

		################

		cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_vr[full_per_attack_vr > 0], make_plot=False, n_sensors=n_cols))
		
		if with_graphs:
			cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_nr[full_per_attack_nr > 0], make_plot=False, n_sensors=n_cols))

		total_attacks = full_per_attack_vr.shape[0]

		print('==========')
		print(f'For {dict_header}')
		print('==========')
		print(f'Total detected: {np.sum(full_per_attack_vr > 0)} / {total_attacks}')
		print(f'Total rank == 1: {np.sum(full_per_attack_vr == 1)} / {total_attacks}')
		print(f'Total graphrank == 1: {np.sum(full_per_attack_nr == 1)} / {total_attacks}')

	if with_graphs:
		labels = [
				'SM', 'SM+', 'SG', 'SG+', 'IG', 'IG+', 'EG', 'EG+',
				'LIME', 'LIME+', 'SHAP', 'SHAP+', 'LEMNA', 'LEMNA+',
				'CF-Add', 'CF-Add+', 'CF-Sub', 'CF-Sub+', 
				'MSE', 'MSE+']
		styles = ['-', 'dashed'] * len(input_config)
	else:
		labels = [
				'SM', 'SG', 'IG', 'EG', 
				'LIME', 'SHAP', 'LEMNA',
				'CF-Add', 'CF-Sub', 
				'MSE']
		styles = ['-'] * len(input_config)

	## AUCs
	print(f'Baseline: {tep_plot_utils.get_auc(np.arange(n_cols + 1) / n_cols, np.arange(n_cols + 1) / n_cols)}')
	for obj_i in range(len(cdf_objs)):
		auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
		print(f'For {labels[obj_i]}, {auc_score}')

	if plot:

		fig, ax = plt.subplots(1, 1, figsize=(8, 6))

		baseline_x = np.arange(0, 1.1, 0.1)
		baseline_values = np.zeros(11)
		for i in range(11):
			baseline_values[i] = n_cols * baseline_x[i]

		ax.plot(baseline_values, baseline_x, color='black', linestyle='dashed')

		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti], linestyle=styles[ploti])

		ax.set_yticks(np.arange(0, 1.1, 0.1))
		ax.grid(which='minor', axis='y')

		ax.set_ylim([0, 1.05])
		ax.set_xlim([0, n_cols])
		ax.set_xlabel('# of Features Examined', fontsize=20)
		ax.set_ylabel('% of Attacks Explained', fontsize=20)

		ax.legend(loc='lower right', fontsize=16, ncol=2)
		fig.tight_layout()

		if with_graphs:
			plt.savefig(f'explanations-dir/plot-attacks-baseline-cdfs-graphs.pdf')
		else:
			plt.savefig(f'explanations-dir/plot-attacks-baseline-cdfs.pdf')
		
		plt.close()

	print('==============================')

def plot_cdf_averaging(swat_filename, tepk_filename):

	graph_suffix = 'filtered'
	swat_graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-SWAT-CLEAN-{graph_suffix}.gml')
	tepk_graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEPK-{graph_suffix}.gml')
	n_cols = len(sensor_cols['SWAT-CLEAN'])

	cdf_objs = []
	configs = [
		{'selection' : 0 , 'averaging': False},
		{'selection' : 49 , 'averaging': False},
		{'selection' : 49 , 'averaging': True},
		{'selection' : 99 , 'averaging': False},
		{'selection' : 99 , 'averaging': True}
	]

	dict_header = swat_filename[:-4]

	for config in configs:

		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(swat_filename,
				'SWAT-CLEAN',
				graph=swat_graph,
				selection_idx=config['selection'],
				use_averaging=config['averaging'],
				verbose=-1)

		_, per_attack_vr2, per_attack_nr2 = open_and_explore_graphrank(tepk_filename,
				'TEPK',
				graph=tepk_graph,
				selection_idx=config['selection'],
				use_averaging=config['averaging'],
				verbose=-1)

		# Normalize rankings such that they are same length
		per_attack_vr2 = (np.array(per_attack_vr2) / 53) * n_cols
		per_attack_nr2 = (np.array(per_attack_nr2) / 53) * n_cols

		full_per_attack_vr = np.concatenate([per_attack_vr, per_attack_vr2])
		full_per_attack_nr = np.concatenate([per_attack_nr, per_attack_nr2])

		cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_vr[full_per_attack_vr > 0], make_plot=False, n_sensors=n_cols))
		cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_nr[full_per_attack_nr > 0], make_plot=False, n_sensors=n_cols))

	labels = ['t1', 't1+GraphWeight', 't50', 't50+GraphWeight', 'avg50', 'avg50-GraphWeight', 't100', 't100+GraphWeight', 'avg100', 'avg100-GraphWeight']
	styles = ['-', 'dashed'] * 5

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	for ploti in range(len(cdf_objs)):
		ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti//2], linestyle=styles[ploti])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1.05])
	ax.set_xlim([0, n_cols])
	ax.set_xlabel('# of Features Examined')
	ax.set_ylabel('% of attacks explained')

	ax.legend()
	fig.tight_layout()
	plt.savefig(f'explanations-dir/plot-attacks-averaging-cdfs-{dict_header}.pdf')
	plt.close()

	for obj_i in range(len(cdf_objs)):
		auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
		print(f'For {dict_header} {labels[obj_i]}, {auc_score}')

def plot_cdf_averaging_slice(input_config):

	graph_suffix = 'filtered'
	swat_graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-SWAT-CLEAN-{graph_suffix}.gml')
	tepk_graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEPK-{graph_suffix}.gml')
	n_cols = len(sensor_cols['SWAT-CLEAN'])

	plot_configs = [
		#{'selection' : 0 , 'summary': None, 'graphweight': True },
		{'selection' : 0 , 'summary': None, 'graphweight': False },

		# {'selection' : 49 , 'summary': None, 'graphweight': True },
		# {'selection' : 49 , 'summary': None, 'graphweight': False },
		{'selection' : 49 , 'summary': 'mean', 'graphweight': True },
		#{'selection' : 49 , 'summary': 'mean', 'graphweight': False },

		# {'selection' : 99 , 'summary': None, 'graphweight': True },
		# {'selection' : 99 , 'summary': None, 'graphweight': False },
		{'selection' : 99 , 'summary': 'mean', 'graphweight': True },
		#{'selection' : 99 , 'summary': 'mean', 'graphweight': False },

	]

	### Score baseline
	_, per_attack_vr, _ = open_and_explore_graphrank('mserank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl',
		'SWAT-CLEAN',
		graph=None,
		selection_idx=0,
		use_averaging=False,
		verbose=-1)

	_, per_attack_vr2, _ = open_and_explore_graphrank('mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl',
		'TEPK',
		graph=None,
		selection_idx=0,
		use_averaging=False,
		verbose=-1)

	per_attack_vr = np.array(per_attack_vr)
	per_attack_vr2 = (np.array(per_attack_vr2) / 53) * n_cols
	baseline_per_attack_vr = np.concatenate([per_attack_vr, per_attack_vr2])
	mse_baseline_cdf = tep_plot_utils.plot_cdf(baseline_per_attack_vr[baseline_per_attack_vr > 0], make_plot=False, n_sensors=n_cols)
	##############

	for plot_config in plot_configs:

		cdf_objs = []

		selection_idx = plot_config['selection']
		summary_mode = plot_config['summary']
	
		use_averaging = (summary_mode == 'mean')
		use_graph = plot_config['graphweight']

		for config in input_config:

			_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['swat-filename'],
				'SWAT-CLEAN',
				graph=swat_graph,
				selection_idx=selection_idx,
				use_averaging=use_averaging,
				verbose=-1)

			_, per_attack_vr2, per_attack_nr2 = open_and_explore_graphrank(config['tepk-filename'],
				'TEPK',
				graph=tepk_graph,
				selection_idx=selection_idx,
				use_averaging=use_averaging,
				verbose=-1)

			if use_graph:
				per_attack_nr2 = (np.array(per_attack_nr2) / 53) * n_cols
				full_per_attack_nr = np.concatenate([per_attack_nr, per_attack_nr2])
				cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_nr[full_per_attack_nr > 0], make_plot=False, n_sensors=n_cols))
			else:
				per_attack_vr2 = (np.array(per_attack_vr2) / 53) * n_cols
				full_per_attack_vr = np.concatenate([per_attack_vr, per_attack_vr2])
				cdf_objs.append(tep_plot_utils.plot_cdf(full_per_attack_vr[full_per_attack_vr > 0], make_plot=False, n_sensors=n_cols))

		labels = ['SM', 'SG', 'IG', 'EG', 'LEMNA', 'MSE']
		styles = ['-'] * len(cdf_objs)

		fig, ax = plt.subplots(1, 1, figsize=(8, 6))
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti], linestyle=styles[ploti])

		ax.plot(mse_baseline_cdf[:, 0], mse_baseline_cdf[:, 1], label='MSE-baseline', color=normal12[5], linestyle='dashed')
		ax.set_yticks(np.arange(0, 1.1, 0.1))
		ax.grid(which='minor', axis='y')

		ax.set_ylim([0, 1.05])
		ax.set_xlim([0, 35])
		ax.set_xlabel('# of Features Examined', fontsize=20)
		ax.set_ylabel('% of Attacks Explained', fontsize=20)

		ax.legend(loc='lower right', fontsize=20)
		fig.tight_layout()

		if summary_mode is not None and use_graph:
			plt.savefig(f'explanations-dir/plot-att-cdfs-{summary_mode}{selection_idx+1}-graphweight.pdf')
		elif summary_mode is not None:
			plt.savefig(f'explanations-dir/plot-att-cdfs-{summary_mode}{selection_idx+1}.pdf')
		elif use_graph:
			plt.savefig(f'explanations-dir/plot-att-cdfs-t{selection_idx+1}-graphweight.pdf')
		else:
			plt.savefig(f'explanations-dir/plot-att-cdfs-t{selection_idx+1}.pdf')

		plt.close()

		for obj_i in range(len(cdf_objs)):
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {summary_mode}{selection_idx} graph{use_graph} {labels[obj_i]}, {auc_score}')

def plot_magnitude_scatter(input_configs):

	plot_z = 0

	for config in input_configs:

		dict_header = config['swat-filename'][:-4]
		scatter_obj_x = []
		scatter_obj_y = []

		_, per_attack_vr, _ = open_and_explore_graphrank(config['swat-filename'],
			'SWAT-CLEAN',
			graph=None,
			verbose=-1)

		per_attack_vr = np.array(per_attack_vr)

		################

		sd_obj = attack_utils.get_attack_sds('SWAT-CLEAN')
		for i in range(len(sd_obj)):
			
			#if per_attack_vr[i] > 0:
			scatter_obj_x.append(np.abs(sd_obj[i][3]))
			scatter_obj_y.append(np.ceil(per_attack_vr[i]))

		_, per_attack_vr2, _ = open_and_explore_graphrank(config['tepk-filename'],
			'TEPK',
			graph=None,
			verbose=-1)
		per_attack_vr2 = np.array(per_attack_vr2)

		sd_obj2 = attack_utils.get_attack_sds('TEPK')
		for i in range(len(sd_obj2)):
			
			#if per_attack_vr2[i] > 0:
			scatter_obj_x.append(np.abs(sd_obj2[i][3]))
			scatter_obj_y.append(np.ceil(per_attack_vr2[i]))

		fig, ax = plt.subplots(1, 1, figsize=(8, 6))

		# Example for plot
		ax.vlines(x=2, ymin=0, ymax=35, linestyles='dashed')
		ax.hlines(y=5, xmin=0, xmax=50, linestyles=':')

		ax.scatter(scatter_obj_x, scatter_obj_y)
		ax.set_xlabel('Attack Magnitude (SDs)', fontsize=20)
		ax.set_ylabel('TopK ranking', fontsize=20)

		#ax.set_xlim([0, 50])
		#ax.set_ylim([1, 35])

		fig.tight_layout()
		plt.savefig(f'explanations-dir/plot-all-scatter-magnitude-{dict_header}.pdf')
		plt.close()

		plot_z += 1

if __name__ == '__main__':

	# import os
	# os.chdir('..')

	###################################################
	### For baseline explanations
	###################################################

	input_config = [
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-saliency_map_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-saliency_map_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-saliency_map_mse_history-tp100.pkl', 
			'suffix': 'in' },

		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-smooth_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-smooth_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-smooth_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-integrated_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-integrated_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-integrated_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-expected_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-expected_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-expected_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },

		{ 'swat-filename': 'limerank-SWAT-CLEAN-default-perfeat-summarized.pkl', 'tepk-filename': 'limerank-TEPK-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		{ 'swat-filename': 'shaprank-SWAT-CLEAN-default-perfeat-summarized.pkl', 'tepk-filename': 'shaprank-TEPK-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		{ 'swat-filename': 'lemnarank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 'tepk-filename': 'lemnarank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		
		# { 'swat-filename': 'cfrank-SWAT-CLEAN-w10-perfeat-tp100.pkl', 'tepk-filename': 'cfrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		# { 'swat-filename': 'cfminrank-SWAT-CLEAN-w10-perfeat-tp100.pkl', 'tepk-filename': 'cfminrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		# { 'swat-filename': 'mserank-SWAT-CLEAN-w10-perfeat-tp100.pkl', 'tepk-filename': 'mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },

		{ 'swat-filename': 'cfrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 'tepk-filename': 'cfrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{ 'swat-filename': 'cfminrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 'tepk-filename': 'cfminrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{ 'swat-filename': 'mserank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 'tepk-filename': 'mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },

	]

	#plot_original_grouped_cdf(input_config, with_graphs=False, plot=False)
	plot_magnitude_scatter([{ 'swat-filename': 'mserank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 'tepk-filename': 'mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' }])

	avg_config = [
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-saliency_map_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-saliency_map_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-saliency_map_mse_history-tp100.pkl', 
			'suffix': 'in' },

		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-smooth_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-smooth_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-smooth_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-integrated_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-integrated_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-integrated_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },
		
		{ 	#'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-expected_gradients_mse_history-tp100.pkl', 
			'swat-filename': 'gradrank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-w10-perfeat-expected_gradients_mse_history-tp100.pkl', 
			'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-expected_gradients_mse_history-tp100.pkl', 
			'suffix': 'in' },

		{ 	'swat-filename': 'lemnarank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 
			'tepk-filename': 'lemnarank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		
		{ 	'swat-filename': 'mserank-CNN-SWAT-CLEAN-l2-hist200-kern3-units32-default-perfeat-tp100.pkl', 
			'tepk-filename': 'mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },

	]

	# avg_config = [
	# 	{ 	'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-saliency_map_mse_history-tp100.pkl', 
	# 		'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-saliency_map_mse_history-tp100.pkl', 
	# 		'suffix': 'in' },
		
	# 	{ 	'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-smooth_gradients_mse_history-tp100.pkl', 
	# 		'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-smooth_gradients_mse_history-tp100.pkl', 
	# 		'suffix': 'in' },
		
	# 	{ 	'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-integrated_gradients_mse_history-tp100.pkl', 
	# 		'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-integrated_gradients_mse_history-tp100.pkl', 
	# 		'suffix': 'in' },
		
	# 	{ 	'swat-filename': 'gradrank-SWAT-CLEAN-perfeat-w10-expected_gradients_mse_history-tp100.pkl', 
	# 		'tepk-filename': 'gradrank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-expected_gradients_mse_history-tp100.pkl', 
	# 		'suffix': 'in' },

	# 	{ 	'swat-filename': 'lemnarank-CNN-SWAT-CLEAN-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 
	# 	  	'tepk-filename': 'lemnarank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl',
	# 	  	'suffix': 'in' },

	# 	{ 	'swat-filename': 'mserank-SWAT-CLEAN-w10-perfeat-tp100.pkl', 
	# 		'tepk-filename': 'mserank-CNN-TEPK-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 
	# 		'suffix': 'in' },
		
	# ]

	plot_cdf_averaging_slice(avg_config)

	# for file_config in avg_config:
	# 	plot_cdf_averaging(file_config['swat-filename'], file_config['tepk-filename'])
	

