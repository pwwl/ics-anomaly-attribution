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

import scipy.stats as ss
import statsmodels.api as sm

np.set_printoptions(suppress=True)
plt.style.use('ggplot')
DEFAULT_CMAP = plt.get_cmap('Reds', 5)
att_skip_list = get_skip_list()

def create_stats_obj(input_config):

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

		_, per_attack_vr_avg, per_attack_nr_avg = open_and_explore_graphrank(
			config['filename'],
			types_to_iterate='all',
			patterns_to_iterate='all',
			suffix=config['suffix'],
			selection_idx=99,
			summary_mode='mean',
			verbose=-1)

		all_attacks = get_non_pid() + get_pid() + get_xmv()
		yticklabels = tep_plot_utils.get_attack_ticklabels()

		df = pd.DataFrame()

		for obji in range(len(all_attacks)):
			for objj in range(len(yticklabels)):
				
				attack_pattern = yticklabels[objj].split('_')[0]
				attack_mag = yticklabels[objj].split('_')[1]
				
				if all_attacks[obji] in get_non_pid():
					feature_type = 'Out'
				elif all_attacks[obji] in get_pid():
					feature_type = 'Sensor'
				elif all_attacks[obji] in get_xmv():
					feature_type = 'Actuator'

				obj = {
					'location': all_attacks[obji],
					'type': feature_type,
					'pattern': attack_pattern,
					'magnitude': attack_mag,
					'mse_rank': per_attack_vr[obji, objj],
					'graph_rank': per_attack_nr[obji, objj],
					'mse_rank_avg': per_attack_vr_avg[obji, objj],
					'graph_rank_avg': per_attack_nr_avg[obji, objj],
				}

				df = df.append(obj, ignore_index=True)

		df.to_csv(f'attack_properties_TEP_{technique}.csv')
		pa_idx_z += 1

def main_stats_obj(input_config):

	pa_idx_z = 0
	techniques = ['SMap', 'SmGrad', 'IG', 'EG', 'LIME', 'SHAP', 'LEMNA', 'CF-Add', 'CF-Sub', 'MSE']
	techniques = ['SMap', 'LEMNA', 'MSE']

	MSE_RANK = 4
	GR_RANK = 0
	GR_RANK_AVG = 1

	for technique in techniques:

		df = pd.read_csv(f'attack_properties_TEP_{technique}.csv', index_col=0)
		df = df[df['mse_rank'] > 0]

		# Filter out non-impactful
		df = df[df['type'] != 'Out']

		print(f'{technique} scores vs graphrank: {np.mean(df.iloc[:, MSE_RANK])} vs GR {np.mean(df.iloc[:, GR_RANK])}')
		stat, pval = ss.f_oneway(df.iloc[:, MSE_RANK], df.iloc[:, GR_RANK])
		print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		print(f'{technique} scores vs graphrank avg: {np.mean(df.iloc[:, MSE_RANK])} vs GR {np.mean(df.iloc[:, GR_RANK_AVG])}')
		stat, pval = ss.f_oneway(df.iloc[:, MSE_RANK], df.iloc[:, GR_RANK_AVG])
		print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		# print('==========')

		# types = ['Sensor', 'Actuator']

		# for sentype in types:
		# 	filter_idx = np.where(df['type'] == sentype)[0]
		# 	print(f'{sentype} {technique} scores vs graphrank: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])}')
		# 	stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK])
		# 	print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		# print('==========')

		# patterns = ['cnst', 'csum', 'line', 'lsum']

		# for pat in patterns:
		# 	filter_idx = np.where(df['pattern'] == pat)[0]
		# 	print(f'{pat} {technique} scores vs graphrank: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])}')
		# 	stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK])
		# 	print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		# print('==========')

		# mags = ['m2s', 'p2s', 'p3s', 'p5s']

		# for mag in mags:
		# 	filter_idx = np.where(df['magnitude'] == mag)[0]
		# 	print(f'{mag} {technique} scores vs graphrank: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])}')
		# 	stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK])
		# 	print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		print('==========')
		print('==========')

		print('==========')

		types = ['Sensor', 'Actuator', 'Out']

		for sentype in types:
			filter_idx = np.where(df['type'] == sentype)[0]
			print(f'{sentype} {technique} scores vs graphrank avg: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])} vs GRavg {np.mean(df.iloc[filter_idx, GR_RANK_AVG])}')
			stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK_AVG])
			print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		print('==========')

		patterns = ['cnst', 'csum', 'line', 'lsum']

		for pat in patterns:
			filter_idx = np.where(df['pattern'] == pat)[0]
			print(f'{pat} {technique} scores vs graphrank avg: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])} vs GRavg {np.mean(df.iloc[filter_idx, GR_RANK_AVG])}')
			stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK_AVG])
			print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		print('==========')

		mags = ['m2s', 'p2s', 'p3s', 'p5s']

		for mag in mags:
			filter_idx = np.where(df['magnitude'] == mag)[0]
			print(f'{mag} {technique} scores vs graphrank avg: {np.mean(df.iloc[filter_idx, MSE_RANK])} vs GR {np.mean(df.iloc[filter_idx, GR_RANK])} vs GRavg {np.mean(df.iloc[filter_idx, GR_RANK_AVG])}')
			stat, pval = ss.f_oneway(df.iloc[filter_idx, MSE_RANK], df.iloc[filter_idx, GR_RANK_AVG])
			print(f'ANOVA={stat:.3f} pval={pval:.5f}')

		print('==========')
		print('==========')

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
		{ 'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
	]

	#create_stats_obj(input_config)
	main_stats_obj(input_config)