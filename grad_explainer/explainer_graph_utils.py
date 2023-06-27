import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pdb
import sys
sys.path.append('..')

from data_loader import load_train_data, load_test_data
from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list
import tep_utils

def build_tep_plc_graph(save_graph=False):

	Xtrain, sensor_cols = load_train_data('TEP')

	###############################
	### Sample covariance
	###############################

	# Samples every 10th training example, finds covariance
	cov = np.cov(Xtrain[::10].T)

	###############################
	### Build graph nodes
	###############################
	G = nx.DiGraph()

	nodes = []
	sensors = []
	xmvs = []

	for i in range(41):
		nodes.append(f's{i+1}')
		sensors.append(f's{i+1}')

	for i in range(12):
		nodes.append(f'a{i+1}')
		xmvs.append(f'a{i+1}')

	G.add_nodes_from(nodes)

	###############################
	### Build graph edges from subprocesses
	###############################
	for sensor_set in tep_utils.FEATURE_SETS:
		for source in sensor_set:
			for dest in sensor_set:

				sidx = sen_to_idx(source)
				didx = sen_to_idx(dest)

				if sidx == didx:
					continue

				edge_weight = np.abs(cov[sidx][didx])

				if edge_weight == 0:
					print(f'Skipping 0 covariance edge: {sensor_cols[sidx]} to {sensor_cols[didx]}')
					continue

				G.add_edge(source, dest, weight=np.abs(cov[sidx][didx]))
				print(f'{sensor_cols[sidx]} flows into {sensor_cols[didx]}')

	###############################
	### Build graph edges from PLCs
	###############################
	for source, dest in tep_utils.PID_EDGES:

		sidx = sen_to_idx(source)
		didx = sen_to_idx(dest)

		edge_weight = np.abs(cov[sidx][didx])

		if edge_weight == 0:
			print(f'Skipping 0 covariance edge: {sensor_cols[sidx]} to {sensor_cols[didx]}')
			continue

		G.add_edge(source, dest, weight=np.abs(cov[sidx][didx]))
		print(f'{sensor_cols[sidx]} flows into {sensor_cols[didx]}')

	if save_graph:
		nx.write_gml(G, f"explanations-dir/graph-TEP-plcs.gml")
		H = convert_tep_to_tepk_graph(G, nodes)
		nx.write_gml(H, f"explanations-dir/graph-TEPK-plcs.gml")

	return G

def build_tep_graph(save_graph=False):

	Xtrain, sensor_cols = load_train_data('TEP')

	###############################
	### Sample covariance
	###############################

	# Samples every 10th training example, finds covariance
	cov = np.cov(Xtrain[::10].T)

	###############################
	### Build graph nodes
	###############################
	G = nx.DiGraph()

	nodes = []
	sensors = []
	xmvs = []

	for i in range(41):
		nodes.append(f's{i+1}')
		sensors.append(f's{i+1}')

	for i in range(12):
		nodes.append(f'a{i+1}')
		xmvs.append(f'a{i+1}')

	G.add_nodes_from(nodes)
	all_edges = tep_utils.PID_EDGES + tep_utils.PROC_EDGES

	for source, dest in all_edges:

		sidx = sen_to_idx(source)
		didx = sen_to_idx(dest)

		edge_weight = np.abs(cov[sidx][didx])

		if edge_weight == 0:
			print(f'Skipping 0 covariance edge: {sensor_cols[sidx]} to {sensor_cols[didx]}')
			continue

		G.add_edge(source, dest, weight=np.abs(cov[sidx][didx]))
		print(f'{sensor_cols[sidx]} flows into {sensor_cols[didx]}')

	if save_graph:
		nx.write_gml(G, f"explanations-dir/graph-TEP-filtered.gml")
		H = convert_tep_to_tepk_graph(G, nodes)
		nx.write_gml(H, f"explanations-dir/graph-TEPK-filtered.gml")

	return G

def convert_tep_to_tepk_graph(G, nodes):

	_, sensor_cols_k = load_train_data('TEPK')

	# For legacy reasons, the TEP graph needs s1, s2 style node labels. 
	# But the TEPK graph needs the sensor column name itself, similar to SWaT
	# TODO: fix this
	mapping = dict()
	for i in range(len(nodes)):
		mapping[nodes[i]] = sensor_cols_k[i]
	H = nx.relabel_nodes(G, mapping, copy=True)

	return H

def plot_tep_graph(graph, positions=None, input_scores=None, name='tep-graph.pdf'):
	
	node_list = list(graph.nodes)

	if positions is None:
		positions = nx.spring_layout(graph, k=0.5)
		
	nx.draw_networkx_edges(graph, positions, 
		arrows=True, 
		arrowsize=5)
	nx.draw_networkx_labels(graph, positions, font_size=4)

	if input_scores is None:
		nx.draw_networkx_nodes(graph, positions, 
			nodelist=node_list)
	else:
		nc = nx.draw_networkx_nodes(graph, positions, 
			nodelist=node_list,
			node_color=input_scores, 
			node_size=200, 
			vmin=0,
			vmax=0.25,
			cmap=mpl.cm.coolwarm)
		plt.colorbar(nc)

	plt.axis('off')
	plt.savefig(name)
	plt.close()

	return 

def get_hardcoded_positions():

	gpp_dict = dict()
	gpp_dict['a3'] = (0, 10)
	gpp_dict['s1'] = (1, 10)
	gpp_dict['a1'] = (0, 9)
	gpp_dict['s2'] = (1, 9)
	gpp_dict['a2'] = (0, 8)
	gpp_dict['s3'] = (1, 8)
	gpp_dict['a4'] = (1, 0)
	gpp_dict['s4'] = (0, 0)

	gpp_dict['s20'] = (3, 11)
	gpp_dict['s5'] = (3, 10)

	gpp_dict['s11'] = (6, 6)
	gpp_dict['s12'] = (6.5, 6)
	gpp_dict['s13'] = (7, 6)

	gpp_dict['s6'] = (3, 8)
	gpp_dict['s16'] = (3, 7)

	gpp_dict['s8'] = (4, 7.5)
	gpp_dict['s9'] = (4, 7)

	gpp_dict['a11'] = (4, 11)
	gpp_dict['s22'] = (5, 11.5)

	gpp_dict['s14'] = (4, 4.5)
	gpp_dict['a7'] = (4, 4)

	gpp_dict['s7'] = (5, 10)
	gpp_dict['s21'] = (5, 7.5)
	gpp_dict['a10'] = (5, 7)

	gpp_dict['a6'] = (7.5, 9)
	gpp_dict['s10'] = (8, 7)

	gpp_dict['a8'] = (6, 0.5)
	
	gpp_dict['s17'] = (7.5, 4)

	gpp_dict['a9'] = (1, 2.5)
	gpp_dict['s19'] = (1, 3)

	gpp_dict['s16'] = (2, 3)
	gpp_dict['s15'] = (2, 2.5)
	gpp_dict['s18'] = (2, 2)

	gpp_dict['s23'] = (-1.5, 6.5)
	gpp_dict['s24'] = (-1.5, 6)
	gpp_dict['s25'] = (-1.5, 5.5)
	gpp_dict['s26'] = (-1.5, 5)
	gpp_dict['s27'] = (-1.5, 4.5)
	gpp_dict['s28'] = (-1.5, 4)

	gpp_dict['s29'] = (10, 10)
	gpp_dict['s30'] = (10, 9.5)
	gpp_dict['s31'] = (10, 9)
	gpp_dict['s32'] = (10, 8.5)
	gpp_dict['s33'] = (10, 8)
	gpp_dict['s34'] = (10, 7.5)
	gpp_dict['s35'] = (10, 7)
	gpp_dict['s36'] = (10, 6.5)

	gpp_dict['s37'] = (10, 2)
	gpp_dict['s38'] = (10, 1.5)
	gpp_dict['s39'] = (10, 1)
	gpp_dict['s40'] = (10, 0.5)
	gpp_dict['s41'] = (10, 0)

	gpp_dict['a5'] = (-1, 1)
	gpp_dict['a12'] = (-1, 0)

	return gpp_dict

def graph_propagation(graph, input_scores, use_correlation=False, n_iterations=1):

	adjusted_scores = np.copy(input_scores)

	for i in range(n_iterations):

		for idx in range(len(adjusted_scores)):

			sensor = idx_to_sen(idx)

			# Find the total covariance (for weighted sum)
			total_corr = 1
			for child in graph[sensor].keys():
				total_corr += graph[sensor][child]['weight']

			if use_correlation:
				total_score = adjusted_scores[idx] * (1 / total_corr)
			else:
				total_score = adjusted_scores[idx]

			# Take sum of all children
			for child in graph[sensor].keys():

				quant_score = adjusted_scores[sen_to_idx(child)]
				corr_score = graph[sensor][child]['weight']

				if use_correlation:
					total_score += quant_score * (corr_score / total_corr)
				else:
					total_score += quant_score

			adjusted_scores[idx] = total_score / (len(graph[sensor].keys()) + 1)

	return adjusted_scores

if __name__ == '__main__':
	
	import os
	os.chdir('..')

	build_tep_graph(save_graph=True)
	build_tep_plc_graph(save_graph=True)
	
