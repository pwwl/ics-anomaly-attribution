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

		for graph in graphs:

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
			# all_aucs[cfg_i, plot_z] = auc_score

	# plot_z = 0

	# for graph in graphs:

	# 	print('NEW GRAPH')
	# 	plot_z += 1

	# 	for cfg_i in range(len(input_config)):

	# 		config = input_config[cfg_i]
	# 		dict_header = config["filename"][:-4]
	# 		_, _, per_attack_nr = open_and_explore_graphrank(config['filename'],
	# 				graph=graph,
	# 				suffix=config['suffix'],
	# 				verbose=-1)

	# 		################

	# 		cdf_obj = tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False)
	# 		cdf_objs.append(cdf_obj)
			
	# 		auc_score = tep_plot_utils.get_auc(cdf_obj[:,0], cdf_obj[:,1])
	# 		print(f'For {dict_header}, {auc_score}')
	# 		all_aucs[cfg_i, plot_z] = auc_score

	# labels = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'CF-Add', 'CF-Sub', 'CI', 'MSE']
	# width = 0.18
	# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	# ax.bar(np.arange(len(all_aucs)) - 3/2 * width, all_aucs[:, 0], width=width, label='No graph')
	# ax.bar(np.arange(len(all_aucs)) - 1/2 * width, all_aucs[:, 1], width=width, label='Graph from Process Knowledge')
	# ax.bar(np.arange(len(all_aucs)) + 1/2 * width, all_aucs[:, 2], width=width, label='Graph from PLCs/Subprocesses')
	# ax.bar(np.arange(len(all_aucs)) + 3/2 * width, all_aucs[:, 3], width=width, label='Graph from Simulator')
	
	# ax.legend(fontsize=16)
	# ax.set_ylim([0, 1.1])
	# ax.set_ylabel('AUC', fontsize=24)
	# ax.set_xticks(np.arange(len(all_aucs)))
	# ax.set_xticklabels(labels, fontsize=16)
	# ax.tick_params(axis='y', which='major', labelsize=16)

	# fig.tight_layout()
	# plt.savefig('plot-tep-baseline-graphs-bar.pdf')

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

	# plot_original_grouped_cdf(input_config, with_graphs=False)
	# plot_original_grouped_cdf(input_config, with_graphs=True)
	
	plot_cdf_compare_graphs(input_config)
