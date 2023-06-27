import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pdb
import pickle

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

normal3 = ['#4053d3', '#ddb310', '#b51d14']

def plot_cdf_compare_graph_iterations(filename, graph_suffix, title='MSE'):

	graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-{graph_suffix}.gml')

	dict_header = filename[:-4]
	cdf_objs = []

	for n_iter in range(6):

		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(filename,
				graph=graph,
				patterns_to_iterate='all',
				n_graph_iterations=n_iter,
				verbose=-1)

		cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))

	labels = ['no graph enhancement', '1 iteration', '2 iterations', '3 iterations', '4 iterations', '5 iterations']

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	
	for ploti in range(len(cdf_objs)):
		ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti])
 	
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	# ax.set_title(title, fontsize=20)
	ax.set_ylim([0, 1])
	ax.set_xlim([0, 53])
	ax.set_xlabel('# of Features Examined', fontsize=20)
	ax.set_ylabel('% of attacks explained', fontsize=20)

	ax.legend(loc='lower right', fontsize=20)
	fig.tight_layout()
	plt.savefig(f'explanations-dir/plot-cdfs-{dict_header}-ngraphiter-{graph_suffix}.pdf')
	plt.close()

	for obj_i in range(len(cdf_objs)):
		auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
		print(f'For {dict_header} {labels[obj_i]}, {auc_score}')

def plot_cdf_compare_averaging_amounts(filename, graph_suffix, use_graph=False):

	graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-{graph_suffix}.gml')

	configs = [
		{'selection' : 0 , 'summary': None},
		{'selection' : 49 , 'summary': None},
		{'selection' : 99 , 'summary': None},
		{'selection' : 49 , 'summary': 'mean'},
		{'selection' : 99 , 'summary': 'mean'}
	]

	dict_header = filename[:-4]
	cdf_objs = []

	# Include MSE, no graph as a baseline
	if use_graph:
		
		_, per_attack_vr, _ = open_and_explore_graphrank('mserank-TEP-w10-perfeat-tp100.pkl',
			graph=graph,
			patterns_to_iterate='all',
			selection_idx=0,
			summary_mode=None,
			verbose=-1)

		mse_baseline_cdf = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False)
		cdf_objs.append(mse_baseline_cdf)
		labels = ['t=1', 't=1 graph-enhanced', 't=50 graph-enhanced', 't=100 graph-enhanced', 'avg t=1-50 graph-enhanced', 'avg t=1-100 graph-enhanced']
	else:
		labels = ['t=1', 't=50', 't=100', 'avg t=1-50', 'avg t=1-100']

	for config in configs:

		_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(filename,
				graph=graph,
				selection_idx=config['selection'],
				summary_mode=config['summary'],
				patterns_to_iterate='all',
				verbose=-1)

		################

		if use_graph:
			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))
		else:
			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False))
	
	styles = ['-', '-', '-', '-', 'dashed', 'dashed']
	colors = [normal12[0], normal12[1], normal12[2], normal12[3], normal12[2], normal12[3]]

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	
	for ploti in range(len(cdf_objs)):
		ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=colors[ploti], linestyle=styles[ploti])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1])
	ax.set_xlim([0, 53])
	ax.set_xlabel('# of Features Examined', fontsize=20)
	ax.set_ylabel('% of attacks explained', fontsize=20)

	ax.legend(loc='lower right', fontsize=20)
	fig.tight_layout()

	if use_graph:
		plt.savefig(f'explanations-dir/plot-cdfs-{dict_header}-average-graph-{graph_suffix}.pdf')
	else:
		plt.savefig(f'explanations-dir/plot-cdfs-{dict_header}-average.pdf')
	
	plt.close()

	for obj_i in range(len(cdf_objs)):
		auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
		print(f'For {dict_header} {labels[obj_i]}, {auc_score}')

	return

def plot_cdf_averaging_slice(input_config, graph_suffix):

	graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-{graph_suffix}.gml')

	plot_configs = [
		{'selection' : 0 , 'summary': None, 'graphweight': False },
		{'selection' : 0 , 'summary': None, 'graphweight': True },
		
		{'selection' : 49 , 'summary': None, 'graphweight': True },
		{'selection' : 49 , 'summary': None, 'graphweight': False },
		{'selection' : 49 , 'summary': 'mean', 'graphweight': False },
		{'selection' : 49 , 'summary': 'mean', 'graphweight': True },

		{'selection' : 99 , 'summary': None, 'graphweight': True },
		{'selection' : 99 , 'summary': None, 'graphweight': False },
		{'selection' : 99 , 'summary': 'mean', 'graphweight': False },
		{'selection' : 99 , 'summary': 'mean', 'graphweight': True },

	]

	_, per_attack_vr, per_attack_nr = open_and_explore_graphrank('mserank-TEP-w10-perfeat-tp100.pkl',
		graph=graph,
		patterns_to_iterate='all',
		selection_idx=0,
		summary_mode=None,
		verbose=-1)

	mse_baseline_cdf = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False)

	for plot_config in plot_configs:

		cdf_objs = []

		selection_idx = plot_config['selection']
		summary_mode = plot_config['summary']
		use_graph = plot_config['graphweight']

		for config in input_config:

			_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['filename'],
					graph=graph,
					patterns_to_iterate='all',
					selection_idx=selection_idx,
					summary_mode=summary_mode,
					verbose=-1)

			################

			if use_graph:
				cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))
			else:
				cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False))

		labels = ['SM', 'SG', 'IG', 'EG', 'LEMNA', 'MSE']
		styles = ['-'] * len(cdf_objs)

		fig, ax = plt.subplots(1, 1, figsize=(8, 6))
		for ploti in range(len(cdf_objs)):
			ax.plot(cdf_objs[ploti][:, 0], cdf_objs[ploti][:, 1], label=labels[ploti], color=normal12[ploti], linestyle=styles[ploti])

		ax.plot(mse_baseline_cdf[:, 0], mse_baseline_cdf[:, 1], label='MSE-baseline', color=normal12[5], linestyle='dashed')
		ax.set_yticks(np.arange(0, 1.1, 0.1))
		ax.grid(which='minor', axis='y')

		ax.set_ylim([0, 1])
		ax.set_xlim([0, 53])
		ax.set_xlabel('# of Features Examined', fontsize=20)
		ax.set_ylabel('% of attacks explained', fontsize=20)

		ax.legend(loc='lower right', fontsize=20)
		fig.tight_layout()

		if summary_mode is not None and use_graph:
			plt.savefig(f'explanations-dir/plot-cdfs-{summary_mode}{selection_idx+1}-graphweight-{graph_suffix}.pdf')
		elif summary_mode is not None:
			plt.savefig(f'explanations-dir/plot-cdfs-{summary_mode}{selection_idx+1}.pdf')
		elif use_graph:
			plt.savefig(f'explanations-dir/plot-cdfs-t{selection_idx+1}-graphweight-{graph_suffix}.pdf')
		else:
			plt.savefig(f'explanations-dir/plot-cdfs-t{selection_idx+1}.pdf')

		plt.close()

		for obj_i in range(len(cdf_objs)):
			auc_score = tep_plot_utils.get_auc(cdf_objs[obj_i][:,0], cdf_objs[obj_i][:,1])
			print(f'For {summary_mode}{selection_idx}, graph{use_graph} {labels[obj_i]}, {auc_score}')

		auc_score = tep_plot_utils.get_auc(mse_baseline_cdf[:,0], mse_baseline_cdf[:,1])
		print(f'For {summary_mode}{selection_idx} graph{use_graph} MSE-baseline, {auc_score}')

		pdb.set_trace()

if __name__ == '__main__':

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
		{ 'filename': 'lemnarank-CNN-TEP-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{ 'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
	]

	# for plot_i in range(len(input_config)):

	# 	filename = input_config[plot_i]['filename']
	# 	#plot_cdf_compare_averaging_amounts(filename, use_graph=False)
	# 	plot_cdf_compare_averaging_amounts(filename, 'filtered', use_graph=True)
	# 	#plot_cdf_compare_graph_iterations(filename, 'filtered')
	# 	#plot_cdf_compare_averaging_amounts(filename, 'plcs', use_graph=True)
		
	plot_cdf_averaging_slice(input_config, 'filtered')
	
	# print('=====================')
	# plot_cdf_averaging_slice(input_config, 'plcs')
