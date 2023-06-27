import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pdb
import pickle

# Internal imports
import sys
sys.path.append('..')

import tep_utils
import tep_plot_utils
from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, scores_to_rank, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list
from main_tep_score_manipulations import open_and_explore_graphrank

np.set_printoptions(suppress=True)
plt.style.use('ggplot')
DEFAULT_CMAP = plt.get_cmap('Reds', 5)
att_skip_list = get_skip_list()

normal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
for i in range(12):
	(c1, c2, c3) = normal12[i]
	normal12[i] = (c1 / 255, c2 / 255, c3 / 255)

def plot_baseline_explanation_heatmap(input_config):

	all_attacks = get_non_pid() + get_pid() + get_xmv()
	yticklabels = tep_plot_utils.get_attack_ticklabels()

	all_attack_len = len(all_attacks)
	per_attack_stories = np.zeros((len(input_config), all_attack_len, len(yticklabels)))
	total_number_attacks = (all_attack_len * len(yticklabels)) - len(tep_utils.get_skip_list())

	pa_idx_z = 0

	for config in input_config:
		print('==========')
		print(f'For {config["filename"]} with {config["suffix"]}')
		print('==========')
		plot_header = config["filename"][:-4]

		per_attack_baseline, _, _ = open_and_explore_graphrank(config['filename'],
			types_to_iterate=['p2s', 'm2s', 'p3s', 'p5s'],
			suffix=config['suffix'],
			verbose=-1)

		per_attack_stories[pa_idx_z] = per_attack_baseline
		pa_idx_z += 1

		explain_outcomes = [np.sum(per_attack_baseline >= 0),
			np.sum(per_attack_baseline == 0),
			np.sum(per_attack_baseline == 1),
			np.sum(per_attack_baseline == 2),
			np.sum(per_attack_baseline == 3)]

		total_attacks = (per_attack_baseline.shape[0] * per_attack_baseline.shape[1]) - len(tep_utils.get_skip_list())

		print(f'Total detected: {explain_outcomes[0]} / {total_attacks}')
		print(f'Total detected but not explained: {explain_outcomes[1]} / {total_attacks}')
		print(f'Total not explained, but graph neighbor: {explain_outcomes[2]} / {total_attacks}')
		print(f'Total explained by graph algorithm: {explain_outcomes[3]} / {total_attacks}')
		print(f'Total explained by MSE: {explain_outcomes[4]} / {total_attacks}')

		plot_baseline = np.zeros_like(per_attack_baseline)

		plot_baseline[per_attack_baseline == -1] = -1
		plot_baseline[per_attack_baseline == 0] = 0
		plot_baseline[per_attack_baseline == 1] = 0
		plot_baseline[per_attack_baseline == 2] = 0
		plot_baseline[per_attack_baseline == 3] = 1

		fig, ax = plt.subplots(1, 1, figsize=(20,4))

		cbar_ax = sns.heatmap(plot_baseline.T, ax=ax,
			center=0,
			cmap=plt.get_cmap('Greys', 3)
			)

		cbar = cbar_ax.collections[0].colorbar
		cbar.set_ticks([-0.666, 0, 0.666])
		cbar.set_ticklabels(['Not detected', 'Detected but \nmissed explanation', 'Explained'])

		ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
		ax.set_yticklabels(yticklabels, fontsize=16, rotation=0)
		ax.set_xticks([])

		ax.set_xticks(np.arange(len(all_attacks)) + 0.5)
		ax.set_xticklabels(all_attacks, fontsize=16)

		ax.set_title(f'Baseline score match {np.sum(plot_baseline == 1)} / {total_attacks}, detected {np.sum(plot_baseline >= 0)} / {total_attacks}', fontsize=24)

		fig.tight_layout()
		plt.savefig(f'explanations-dir/plot-{plot_header}-msematch.pdf')
		plt.close()

def plot_ranks_heatmap(input_config):

	pa_idx_z = 0
	techniques = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'LEMNA', 'CF-Add', 'CF-Sub', 'MSE']
	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']

	for config in input_config:

		print('==========')
		print(f'For {config["filename"]} with {config["suffix"]}')
		print('==========')
		plot_header = config["filename"][:-4]
		technique = techniques[pa_idx_z]

		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(
			config['filename'],
			types_to_iterate='all',
			patterns_to_iterate='all',
			suffix=config['suffix'],
			# selection_idx=99,
			# summary_mode='mean',
			verbose=-1)

		total_attacks = (per_attack_vr.shape[0] * per_attack_vr.shape[1]) - len(tep_utils.get_skip_list())

		print(f'Total detected: {np.sum(per_attack_vr > 0)}')
		print(f'Total rank == 1: {np.sum(per_attack_vr == 1)} / {total_attacks}')
		print(f'Total graphrank == 1: {np.sum(per_attack_nr == 1)} / {total_attacks}')

		per_attack_vr[per_attack_vr == -1] = 99
		per_attack_vr[per_attack_vr == 0] = 66

		per_attack_nr[per_attack_nr == -1] = 99
		per_attack_nr[per_attack_nr == 0] = 66

		##########################

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

		ax[0].set_title(f'TopK {technique} ranking of manipulated feature', fontsize=20)
		ax[0].set_yticks(np.arange(len(yticklabels)) + 0.5)
		ax[0].set_yticklabels(yticklabels, fontsize=16, rotation=0)
		ax[0].set_xticks([])

		ax[1].set_title(f'TopK {technique} graph-enhanced ranking of manipulated feature', fontsize=20)
		ax[1].set_yticks(np.arange(len(yticklabels)) + 0.5)
		ax[1].set_yticklabels(yticklabels, fontsize=16, rotation=0)

		ax[1].set_xticks(np.arange(len(all_attacks)) + 0.5)
		ax[1].set_xticklabels(all_attacks, fontsize=16)

		#ax[1].annotate('Annotation', xy=(1, 18), ha="center", va="top")

		ax[1].text(1, 18, 'Out-of-loop sensors', fontsize=20)
		ax[1].text(13, 18, 'In-loop sensors', fontsize=20)
		ax[1].text(26, 18, 'Actuators', fontsize=20)

		ax[1].vlines(x=[7, 23], ymin=16, ymax=20, color='black', lw=2, linestyle='--', clip_on=False)

		#ax[1].set_xlabel('Out-of-loop sensors                         In-loop-sensors            Manipulated variables', fontsize=20)

		fig.tight_layout()
		plt.savefig(f'explanations-dir/plot-{plot_header}-ranks.pdf')
		plt.close()

		pa_idx_z += 1

def plot_latencies(filename, verbose=1):

	latency_results = pickle.load(open(f'explanations-dir/{filename}', 'rb'))

	attacks_to_iterate = get_non_pid() + get_pid() + get_xmv()
	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']
	attack_patterns  = ['cons', 'csum', 'line', 'lsum']

	# Some tricky indexing to build plot object
	per_attack_latency = np.zeros((len(attacks_to_iterate), len(attack_types) * len(attack_patterns)))
	pa_idx_x = -1
	pa_idx_y = -1

	latency_cons = []
	latency_csum = []
	latency_line = []
	latency_lsum = []

	for an in attacks_to_iterate:
		pa_idx_x += 1
		pa_idx_y = -1

		for at in attack_types:
			for ap in attack_patterns:

				pa_idx_y += 1
				attack_footer = f'{ap}_{at}_{an}'

				if attack_footer in att_skip_list:

					if verbose >= 0:
						print(f'Skipping crashing attack {ap} {at} {an}!')
					continue

				latency = latency_results[f'{attack_footer}_time']

				if latency == 0:
					per_attack_latency[pa_idx_x, pa_idx_y] = -1
				else:
					# Adjust for attack start
					per_attack_latency[pa_idx_x, pa_idx_y] = latency - 9955

					if ap == 'cons':
						latency_cons.append(latency - 9955)
					elif ap == 'csum':
						latency_csum.append(latency - 9955)
					elif ap == 'line':
						latency_line.append(latency - 9955)
					elif ap == 'lsum':
						latency_lsum.append(latency - 9955)

	print(f'Average latency on cons: {np.mean(latency_cons)}')
	print(f'Average latency on csum: {np.mean(latency_csum)}')
	print(f'Average latency on constants: {np.mean(latency_cons + latency_csum)}')

	print(f'Average latency on line: {np.mean(latency_line)}')
	print(f'Average latency on lsum: {np.mean(latency_lsum)}')
	print(f'Average latency on linears: {np.mean(latency_line + latency_lsum)}')

	yticklabels = tep_plot_utils.get_attack_ticklabels()
	colormap = sns.color_palette("vlag", as_cmap=True)

	fig, ax = plt.subplots(1, 1, figsize=(16,8))
	cbar_ax = sns.heatmap(per_attack_latency.T, ax=ax,
		annot=per_attack_latency.T,
		fmt='.0f',
		center=0,
		cmap=colormap,
		cbar=False,
		)

	#cbar = cbar_ax.collections[0].colorbar
	#cbar.set_ticks([-0.666, 0, 0.666])
	#cbar.set_ticklabels(['Not detected', 'Detected but \nmissed explanation', 'Explained'])

	ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
	ax.set_yticklabels(yticklabels, fontsize=16, rotation=0)
	ax.set_xticks([])

	ax.set_xticks(np.arange(len(attacks_to_iterate)) + 0.5)
	ax.set_xticklabels(attacks_to_iterate, fontsize=16)

	fig.tight_layout()
	plt.savefig(f'explanations-dir/plot-latency.pdf')
	plt.close()

def plot_original_grouped_cdf(input_config, with_graphs=True):

	graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-filtered.gml')
	cdf_objs = []

	for config in input_config:

		dict_header = config["filename"][:-4]
		print(dict_header)
		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['filename'],
				graph=graph,
				suffix=config['suffix'],
				verbose=-1)

		################

		cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False))
		# print(f'Time {dict_header}: {tep_plot_utils.get_time(per_attack_vr[per_attack_vr >= 1])}')
		# print(f'Time improvement {dict_header}: {tep_plot_utils.get_time_improvement(per_attack_vr[per_attack_vr >= 1])}')

		if with_graphs:
			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))
			# print(f'Time graph-{dict_header}: {tep_plot_utils.get_time(per_attack_nr[per_attack_nr >= 1])}')

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	baseline_x = np.arange(0, 1.1, 0.1)
	baseline_values = np.zeros(11)
	for i in range(11):
		baseline_values[i] = 53 * baseline_x[i]

	ax.plot(baseline_values, baseline_x, color='black', linestyle='dashed')

	if with_graphs:
		labels = ['SMap', 'SMap+', 'SmGrad', 'SmGrad+', 'IG', 'IG+', 'EG', 'EG+',
			'LIME', 'LIME+', 'SHAP', 'SHAP+', 'LEMNA', 'LEMNA+',
			'CF-Add', 'CF-Add+', 'CF-Sub', 'CF-Sub+', 'CI', 'CI+', 'MSE', 'MSE+']
		styles = ['-', 'dashed'] * len(input_config)
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti//2], linestyle=styles[ploti])
	else:
		labels = ['SM', 'SG', 'IG', 'EG', 'LIME', 'SHAP', 'LEMNA', 'CF-Add', 'CF-Sub', 'MSE']
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1])
	ax.set_xlim([0, 53])
	ax.set_xlabel('# of Features Examined', fontsize=16)
	ax.set_ylabel('% of Attacks Explained', fontsize=16)

	ax.legend(loc='lower right', fontsize=16, ncol=2)
	fig.tight_layout()

	if with_graphs:
		plt.savefig('explanations-dir/plot-tep-baseline-cdfs-graphs.pdf')
	else:
		plt.savefig('explanations-dir/plot-tep-baseline-cdfs.pdf')
	plt.close()

	print(f'Baseline: {tep_plot_utils.get_auc(np.arange(54), np.arange(54) / 53)}')

	if with_graphs:
		for obj_i in range(len(cdf_objs)):
			labels = ['SMap', 'SMap+', 'SmGrad', 'SmGrad+', 'IG', 'IG+', 'EG', 'EG+',
				'LIME', 'LIME+', 'SHAP', 'SHAP+', 'LEMNA', 'LEMNA+',
				'CF-Add', 'CF-Add+', 'CF-Sub', 'CF-Sub+', 'CI', 'CI+', 'MSE', 'MSE+']
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {labels[obj_i]}, {auc_score}')
	else:
		for obj_i in range(len(cdf_objs)):
			labels = ['SM', 'SG', 'IG', 'EG', 'LIME', 'SHAP', 'LEMNA', 'CF-Add', 'CF-Sub', 'MSE']
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {labels[obj_i]}, {auc_score}')

	print('==============================')

def plot_cdf_compare_graphs(input_config):

	graphs = [
		nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-filtered.gml'),
		nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-functions.gml'),
		nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-spike50.gml')
		]

	cdf_objs = []
	all_aucs = np.zeros((len(input_config), 1 + len(graphs)))

	# First pass, no graph
	for cfg_i in range(len(input_config)):

		config = input_config[cfg_i]
		dict_header = config["filename"][:-4]
		_, per_attack_vr, _ = open_and_explore_graphrank(config['filename'],
				graph=None,
				suffix=config['suffix'],
				verbose=-1)

		################

		cdf_obj = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False)
		cdf_objs.append(cdf_obj)

		auc_score = tep_plot_utils.get_auc(cdf_obj[:,0], cdf_obj[:,1])
		print(f'For {dict_header}, {auc_score}')

		all_aucs[cfg_i, 0] = auc_score

		#print(f'Time {dict_header}: {tep_plot_utils.get_time(per_attack_vr[per_attack_vr >= 1])}')
		#print(f'Time improvement {dict_header}: {tep_plot_utils.get_time_improvement(per_attack_vr[per_attack_vr >= 1])}')

	plot_z = 0

	for graph in graphs:

		print('NEW GRAPH')
		plot_z += 1

		for cfg_i in range(len(input_config)):

			config = input_config[cfg_i]
			dict_header = config["filename"][:-4]
			_, _, per_attack_nr = open_and_explore_graphrank(config['filename'],
					graph=graph,
					suffix=config['suffix'],
					verbose=-1)

			################

			cdf_obj = tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False)
			cdf_objs.append(cdf_obj)

			auc_score = tep_plot_utils.get_auc(cdf_obj[:,0], cdf_obj[:,1])
			print(f'For {dict_header}, {auc_score}')
			all_aucs[cfg_i, plot_z] = auc_score

			#print(f'Time improvement {dict_header}: {tep_plot_utils.get_time_improvement(per_attack_nr[per_attack_nr >= 1])}')
			#print(f'Time graph-{dict_header}: {tep_plot_utils.get_time(per_attack_nr[per_attack_nr >= 1])}')


	labels = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'CF-Add', 'CF-Sub', 'CI', 'MSE']
	width = 0.18
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.bar(np.arange(len(all_aucs)) - 3/2 * width, all_aucs[:, 0], width=width, label='No graph')
	ax.bar(np.arange(len(all_aucs)) - 1/2 * width, all_aucs[:, 1], width=width, label='Graph from Process Knowledge')
	ax.bar(np.arange(len(all_aucs)) + 1/2 * width, all_aucs[:, 2], width=width, label='Graph from PLCs/Subprocesses')
	ax.bar(np.arange(len(all_aucs)) + 3/2 * width, all_aucs[:, 3], width=width, label='Graph from Simulator')

	ax.legend(fontsize=16)
	ax.set_ylim([0, 1.1])
	ax.set_ylabel('AUC', fontsize=24)
	ax.set_xticks(np.arange(len(all_aucs)))
	ax.set_xticklabels(labels, fontsize=16)
	ax.tick_params(axis='y', which='major', labelsize=16)

	fig.tight_layout()
	plt.savefig('plot-tep-baseline-graphs-bar.pdf')

if __name__ == '__main__':

	# Clean up wd, set options
	import os
	os.chdir('..')

	###################################################
	### For baseline explanations
	###################################################

	input_config = [

		{ 'filename': 'gradrank-TEP-perfeat-w10-saliency_map_mse_history-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'gradrank-TEP-perfeat-w10-smooth_gradients_mse_history-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'gradrank-TEP-perfeat-w10-integrated_gradients_mse_history-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'gradrank-TEP-perfeat-w10-expected_gradients_mse_history-tp100.pkl', 'suffix': 'in' },

		{ 'filename': 'limerank-TEP-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		{ 'filename': 'shaprank-TEP-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		{ 'filename': 'lemnarank-CNN-TEP-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },

		{ 'filename': 'cfrank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'cfminrank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		# { 'filename': 'rulerank-TEP-w1-sdt3-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },

		# { 'filename': 'mserank-TEP-w10-perfeatpost-tp100.pkl', 'suffix': 'post' },

	# 	{ 'filename': 'graphrank-TEP-w1-perfeat.pkl', 'suffix': 'in' },
	#	{ 'filename': 'graphrank-TEP-w10-perfeat.pkl', 'suffix': 'in' },

	# 	{ 'filename': 'graphrank-TEP-w1.pkl', 'suffix': 'mse_in' },
	# 	{ 'filename': 'graphrank-TEP-w10.pkl', 'suffix': 'in' },

	# 	{ 'filename': 'graphrank-TEP-w1-perfeat.pkl', 'suffix': 'post' },
	# 	{ 'filename': 'graphrank-TEP-w10-perfeat.pkl', 'suffix': 'post' },
	# 	{ 'filename': 'graphrank-TEP-w1.pkl', 'suffix': 'post' },
	# 	{ 'filename': 'graphrank-TEP-w9.pkl', 'suffix': 'post' },

	#	{ 'filename': 'rulerank-TEP-w1-sdt3-tp100.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'rulerank-TEP-w9-sdt3.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'rulerank-TEP-w1-sdt3.pkl', 'suffix': 'post' },
	# 	{ 'filename': 'rulerank-TEP-w9-sdt3.pkl', 'suffix': 'post' },

	# 	{ 'filename': 'graphrank-TEP-cherrypick-max.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'graphrank-TEP-cherrypick-mean.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'graphrank-TEP-cherrypick-hist.pkl', 'suffix': 'in' },

	# 	{ 'filename': 'rulerank-TEP-cherrypick-max.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'rulerank-TEP-cherrypick-mean.pkl', 'suffix': 'in' },
	# 	{ 'filename': 'rulerank-TEP-cherrypick-hist.pkl', 'suffix': 'in' },

	]

	#plot_baseline_explanation_heatmap(input_config)
	plot_ranks_heatmap(input_config)
	#plot_latencies('latency-CNN-TEP-l2-hist50-kern3-units64-w10-perfeat.pkl')

	plot_original_grouped_cdf(input_config, with_graphs=False)
	#plot_original_grouped_cdf(input_config, with_graphs=True)

	#plot_cdf_compare_graphs(input_config)
