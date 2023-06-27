import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pdb
import pickle
import sys
import os

sys.path.append('..')

import tep_utils
import tep_plot_utils

from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, scores_to_rank, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list

from main_tep_score_manipulations import open_and_explore_graphrank, aggregate_graphrank

np.set_printoptions(suppress=True)
plt.style.use('ggplot')
DEFAULT_CMAP = plt.get_cmap('Reds', 5)
att_skip_list = get_skip_list()

normal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
for i in range(12):
	(c1, c2, c3) = normal12[i]
	normal12[i] = (c1 / 255, c2 / 255, c3 / 255)

def full_intersections(input_config, topk=10, mode='intersect'):

	mse_preds = aggregate_graphrank(input_config[0]['filename'], use_graph=False, topk=topk)
	mse_graph_preds = aggregate_graphrank(input_config[0]['filename'], use_graph=True, topk=topk)
	mse_avg_preds = aggregate_graphrank(input_config[0]['filename'], selection_idx=49, summary_mode='mean', use_graph=False, topk=topk)
	mse_avg_graph_preds = aggregate_graphrank(input_config[0]['filename'], selection_idx=49, summary_mode='mean', use_graph=True, topk=topk)

	sm_preds = aggregate_graphrank(input_config[1]['filename'], use_graph=False, topk=topk)
	sm_graph_preds = aggregate_graphrank(input_config[1]['filename'], use_graph=True, topk=topk)
	sm_avg_preds = aggregate_graphrank(input_config[1]['filename'], selection_idx=49, summary_mode='mean', use_graph=False, topk=topk)
	sm_avg_graph_preds = aggregate_graphrank(input_config[1]['filename'], selection_idx=49, summary_mode='mean', use_graph=True, topk=topk)

	lm_preds = aggregate_graphrank(input_config[2]['filename'], use_graph=False, topk=topk)
	lm_graph_preds = aggregate_graphrank(input_config[2]['filename'], use_graph=True, topk=topk)
	lm_avg_preds = aggregate_graphrank(input_config[2]['filename'], selection_idx=49, summary_mode='mean', use_graph=False, topk=topk)
	lm_avg_graph_preds = aggregate_graphrank(input_config[2]['filename'], selection_idx=49, summary_mode='mean', use_graph=True, topk=topk)

	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']
	attack_patterns = ['cons', 'csum', 'line', 'lsum']
	all_attacks = get_non_pid() + get_pid() + get_xmv()
	yticklabels = tep_plot_utils.get_attack_ticklabels()
	all_attack_len = len(all_attacks)
	attack_types_len = len(attack_patterns) * len(attack_types)

	intersect_detected_full = np.zeros((all_attack_len, attack_types_len))
	output_set_size = np.zeros((all_attack_len, attack_types_len))

	# Some tricky indexing to build plot object
	pa_idx_x = -1
	pa_idx_y = -1

	for an in all_attacks:
		pa_idx_x += 1
		pa_idx_y = -1

		for at in attack_types:
			
			for ap in attack_patterns:

				pa_idx_y += 1

				attack_footer = f'{ap}_{at}_{an}'
				
				# Attack crashes
				if attack_footer not in mse_preds:
					intersect_detected_full[pa_idx_x, pa_idx_y] = -2
					continue

				# Attack missed
				if mse_preds[attack_footer] is None:
					intersect_detected_full[pa_idx_x, pa_idx_y] = -1
					continue

				attack_target_idx = sen_to_idx(an)
				sets = [mse_avg_graph_preds[attack_footer], mse_preds[attack_footer], lm_avg_preds[attack_footer]]

				if mode == 'intersect':
					output_set = np.arange(54)
					for subset in sets:
						output_set = np.intersect1d(subset, output_set)
				elif mode == 'union':
					output_set = []
					for subset in sets:
						output_set = np.union1d(subset, output_set)
				elif mode == 'vote':
					
					output_set = []
					
					for si in range(53):
					
						votes = 0
						for subset in sets:
							if si in subset:
								votes += 1	

						if votes >= 2:
							output_set.append(si)

				if attack_target_idx in output_set:
					intersect_detected_full[pa_idx_x, pa_idx_y] = 1
				
				output_set_size[pa_idx_x, pa_idx_y] = len(output_set)

	return intersect_detected_full, output_set_size

def plot_intersection(input_config):

	cdf_objs = []

	########## Add TEP MSE baseline
	_, per_attack_vr, per_attack_nr = open_and_explore_graphrank('mserank-TEP-w10-perfeat-tp100.pkl',
		patterns_to_iterate='all',
		selection_idx=0,
		summary_mode=None,
		verbose=-1)

	mse_baseline_cdf = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False)

	########## Add TEP graph averaging baseline

	_, per_attack_vr, per_attack_nr = open_and_explore_graphrank('mserank-TEP-w10-perfeat-tp100.pkl',
		patterns_to_iterate='all',
		selection_idx=49,
		summary_mode='mean',
		verbose=-1)

	tep_best_cdf = tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False)

	########################################

	for sum_mode in ['intersect', 'union', 'vote']:

		plot_obj = np.zeros((54, 2))

		for k in range(54):

			print(f'Measuring: topk={k}')
			detected_full, output_set_size = full_intersections(input_config, topk=k, mode=sum_mode)

			detection_rate = np.mean(detected_full[detected_full >= 0])
			average_size = np.mean(output_set_size[detected_full >= 0])

			print(f'Detect rate: {detection_rate}')
			print(f'Average set size: {average_size}')

			plot_obj[k, 0] = average_size
			plot_obj[k, 1] = detection_rate
			
		plot_obj[0, 0] = 0
		plot_obj[0, 1] = 0
		cdf_objs.append(plot_obj)

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	labels = ['Intersection', 'Union', 'Majority']
	for ploti in range(len(cdf_objs)):
		ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti])
		ax.scatter(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], color=normal12[ploti])

		auc_score = tep_plot_utils.get_auc(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1])
		print(f'{labels[ploti]} AUC: {auc_score}')

	ax.plot(mse_baseline_cdf[:, 0], mse_baseline_cdf[:, 1], label='MSE-baseline', color=normal12[4])
	ax.plot(tep_best_cdf[:, 0], tep_best_cdf[:, 1], label='Previous best', color=normal12[5])

	auc_score = tep_plot_utils.get_auc(mse_baseline_cdf[:, 0], mse_baseline_cdf[:, 1])
	print(f'MSE-baseline AUC: {auc_score}')
	auc_score = tep_plot_utils.get_auc(tep_best_cdf[:, 0], tep_best_cdf[:, 1])
	print(f'Best AUC: {auc_score}')

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1])
	ax.set_xlim([0, 53])
	ax.set_xlabel('Average # of Features Examined', fontsize=20)
	ax.set_ylabel('% of Attacks Explained', fontsize=20)

	ax.legend(loc='lower right', fontsize=20)
	fig.tight_layout()

	plt.savefig('explanations-dir/plot-intersection-ndss.pdf')
	plt.close()

	pdb.set_trace()

def plot_ranks_heatmap(input_config):

	pa_idx_z = 0
	techniques = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'CF-Add', 'CF-Sub', 'CI', 'MSE']
	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']

	for config in input_config:
		
		print('==========')
		print(f'For {config["filename"]} with {config["suffix"]}')
		print('==========')
		plot_header = config["filename"][:-4]
		technique = techniques[pa_idx_z]

		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(
			config['filename'],
			types_to_iterate=attack_types,
			patterns_to_iterate='all',
			suffix=config['suffix'],
			verbose=-1)

		total_attacks = (per_attack_vr.shape[0] * per_attack_vr.shape[1]) - len(tep_utils.get_skip_list())

		print(f'Total rank == 1: {np.sum(per_attack_vr == 1)} / {total_attacks}')
		print(f'Total graphrank == 1: {np.sum(per_attack_nr == 1)} / {total_attacks}')

		per_attack_vr[per_attack_vr == -1] = 99
		per_attack_vr[per_attack_vr == 0] = 99

		per_attack_nr[per_attack_nr == -1] = 99
		per_attack_nr[per_attack_nr == 0] = 99

		########################3

		all_attacks = get_non_pid() + get_pid() + get_xmv()
		yticklabels = tep_plot_utils.get_attack_ticklabels()

		colormap = sns.color_palette("vlag", as_cmap=True)

		fig, ax = plt.subplots(2, 1, figsize=(20,12))
		cbar_ax = sns.heatmap(per_attack_vr.T, ax=ax[0],
			annot=per_attack_vr.T,
			cmap=colormap,
			cbar=False
			)

		sns.heatmap(per_attack_nr.T, ax=ax[1],
			annot=per_attack_nr.T,
			cmap=colormap,
			cbar=False
			)

		ax[0].set_title(f'TopK {technique} ranking of attacked feature', fontsize=20)
		ax[0].set_yticks(np.arange(len(yticklabels)) + 0.5)
		ax[0].set_yticklabels(yticklabels, fontsize=16, rotation=0)
		ax[0].set_xticks([])

		ax[1].set_title(f'TopK {technique} graphweight ranking of attacked feature', fontsize=20)
		ax[1].set_yticks(np.arange(len(yticklabels)) + 0.5)
		ax[1].set_yticklabels(yticklabels, fontsize=16, rotation=0)

		ax[1].set_xticks(np.arange(len(all_attacks)) + 0.5)
		ax[1].set_xticklabels(all_attacks, fontsize=16)

		fig.tight_layout()
		plt.savefig(f'explanations-dir/plot-{plot_header}-ranks.pdf')
		plt.close()

		pa_idx_z += 1

def plot_original_grouped_cdf(input_config, with_graphs=True):

	graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-filtered.gml')
	cdf_objs = []

	for config in input_config:

		dict_header = config["filename"][:-4]
		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['filename'],
				graph=graph,
				suffix=config['suffix'],
				verbose=-1)

		################

		cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False))
		
		if with_graphs:
			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))
		
	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	baseline_x = np.arange(0, 1.1, 0.1)
	baseline_values = np.zeros(11)
	for i in range(11):
		baseline_values[i] = 53 * baseline_x[i]

	ax.plot(baseline_values, baseline_x, color='black', linestyle='dashed')

	if with_graphs:
		labels = ['SMap', 'SMap+', 'SmGrad', 'SmGrad+', 'IG', 'IG+', 'EG', 'EG+', 'LIME', 'LIME+', 'SHAP', 'SHAP+', 
			'CF-Add', 'CF-Add+', 'CF-Sub', 'CF-Sub+', 'CI', 'CI+', 'MSE', 'MSE+']
		styles = ['-', 'dashed'] * len(input_config)
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti//2], linestyle=styles[ploti])
	else:
		labels = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'CF-Add', 'CF-Sub', 'CI', 'MSE']
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1])
	ax.set_xlim([0, 53])
	ax.set_xlabel('# of Features Examined', fontsize=16)
	ax.set_ylabel('% of Attacks Explained', fontsize=16)

	ax.legend(loc='lower right', fontsize=16)
	fig.tight_layout()

	if with_graphs:
		plt.savefig('explanations-dir/plot-tep-baseline-cdfs-graphs.pdf')
	else:
		plt.savefig('explanations-dir/plot-tep-baseline-cdfs.pdf')
	plt.close()

	print(f'Baseline: {tep_plot_utils.get_auc(np.arange(54), np.arange(54) / 53)}')

	if with_graphs:
		for obj_i in range(len(cdf_objs)):
			labels = ['SMap', 'SMap+', 'SmGrad', 'SmGrad+', 'IG', 'IG+', 'EG', 'EG+', 'LIME', 'LIME+', 'SHAP', 'SHAP+', 
				'CF-Add', 'CF-Add+', 'CF-Sub', 'CF-Sub+', 'CI', 'CI+', 'MSE', 'MSE+']
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {labels[obj_i]}, {auc_score}')
	else:
		for obj_i in range(len(cdf_objs)):
			labels = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'CF-Add', 'CF-Sub', 'CI', 'MSE']
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {labels[obj_i]}, {auc_score}')

	print('==============================')

if __name__ == '__main__':

	os.chdir('..')

	###################################################
	### For baseline explanations
	###################################################

	input_config = [
		{ 'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'gradrank-TEP-perfeat-w10-saliency_map_mse_history-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'lemnarank-CNN-TEP-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
	]

	plot_intersection(input_config)
