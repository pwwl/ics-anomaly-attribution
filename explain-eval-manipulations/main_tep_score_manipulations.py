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

import tep_plot_utils
from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, scores_to_rank, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_skip_list
from data_loader import load_train_data

np.set_printoptions(suppress=True)
plt.style.use('ggplot')
DEFAULT_CMAP = plt.get_cmap('Reds', 5)
att_skip_list = get_skip_list()

QUAL_CMAP = ['#003f5c','#2f4b7c','#665191','#a05195','#d45087','#f95d6a','#ff7c43','#ffa600']
#QUAL_CMAP = ['#001482', '#9c007a', '#e60659', '#ff722f', '#ffc400']

def find_and_score_parents(graph, vector_score, nodes_visited, candidate, depth, verbose=0):

	# Keep anything at least 1 SD away
	threshold = 0.5

	# In case of bad infinite loops
	if depth > 4:
		return []

	# Bookkeep nodes
	if candidate in nodes_visited:
		return []

	nodes_visited.append(candidate)
	node_weight = vector_score[sen_to_idx(candidate)]

	if verbose > 0:
		print(f'(depth={depth}) Candidate: {candidate}, {node_weight}: parents {graph.pred[candidate]}')

	if node_weight > threshold:

		has_high_mse_parent = False
		max_parent_score = -1
		max_parent = -1

		for parent in graph.pred[candidate].keys():

			quant_score = vector_score[sen_to_idx(parent)]
			if verbose > 0:
				print(f'Candidate {candidate}, parent {parent}: {quant_score} * {graph.pred[candidate][parent]["weight"]}')

			if quant_score > threshold:
				corr_score = quant_score# * (graph.pred[candidate][parent]['weight'] + 0.01)

				if corr_score > max_parent_score:
					max_parent = parent
					max_parent_score = corr_score

				if verbose > 0:
					print(f'Candidate {candidate} has sufficent parent {parent}: {corr_score}!')

		if max_parent_score > 0:
			return [candidate] + find_and_score_parents(graph, vector_score, nodes_visited, max_parent, depth+1, verbose=verbose)
		else:
			if verbose > 0:
				print(f'Candidate {candidate} kept!')
			return [candidate]

	return []

def full_neighbor_score_sweep(graph, dataset, vector_scores, correlation=False, n_iterations=1):

	neighbor_scores = np.copy(vector_scores)

	for i in range(n_iterations):

		for idx in range(len(neighbor_scores)):

			sensor = idx_to_sen(idx)

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

				quant_score = neighbor_scores[sen_to_idx(child)]
				corr_score = graph[sensor][child]['weight']

				if correlation:
					total_score += quant_score * (corr_score / total_corr)
				else:
					total_score += quant_score

			neighbor_scores[idx] = total_score / (len(graph[sensor].keys()) + 1)

	return neighbor_scores

def find_distance_parents(graph, vector_score, nodes_visited, node_scores, candidate, depth, max_depth=10, verbose=0):

	# In case of bad infinite loops
	if depth > max_depth:
		return dict()

	# Bookkeep nodes, keeping smallest depths
	if candidate in nodes_visited:
		if depth < node_scores[candidate]:
			node_scores[candidate] = depth
		return dict()

	nodes_visited.append(candidate)
	node_scores[candidate] = depth

	node_weight = vector_score[sen_to_idx(candidate)]

	if verbose > 0:
		print(f'(depth={depth}) Candidate: {candidate}, {node_weight}: parents {graph.pred[candidate]}')

	for parent in graph.pred[candidate].keys():

		quant_score = vector_score[sen_to_idx(parent)]
		if verbose > 0:
			print(f'Candidate {candidate}, parent {parent}: {quant_score} * {graph.pred[candidate][parent]["weight"]}')

		find_distance_parents(graph, vector_score, nodes_visited, node_scores, parent, depth+1, verbose=verbose)

	return node_scores

def ranking_to_distance(graph, vector_scores, attack_footer, attack_target, minimum_distance=3, topk_search=3, verbose=0):

	within_k_distance = 0
	full_candidate_set = set()

	sorted_scores_idxs = np.argsort(vector_scores)

	for i in range(topk_search):

		candidate_idx = sorted_scores_idxs[-(i+1)]
		candidate_sensor = idx_to_sen(candidate_idx)
		candidate_chains = find_distance_parents(graph, vector_scores, list(), dict(), candidate_sensor, 0, verbose=verbose)

		for sensor in candidate_chains:
			if candidate_chains[sensor] <= minimum_distance:
				full_candidate_set.add(sensor)

		if attack_target in candidate_chains and candidate_chains[attack_target] <= minimum_distance:
			within_k_distance = 1
			if verbose > 0:
				print(f'For attack {attack_footer}: sensor ranking #{i+1} ({candidate_sensor}) is distance {candidate_chains[attack_target]}')

		if verbose > 0:
			print(f'After using top {i+1}: candidate set is {len(full_candidate_set)}')

	return within_k_distance, len(full_candidate_set)

def top_k_predict(graph, vector_scores, use_graph=True, topk=10, n_graph_iterations=1, verbose=0):

	if use_graph:
		scores = full_neighbor_score_sweep(graph, vector_scores, n_iterations=n_graph_iterations)
	else:
		scores = vector_scores
	
	prediction = np.argsort(scores)[-topk:]

	return prediction

def score_attack_explanation(graph, candidates, vector_scores, attack_footer, attack_target, n_graph_iterations=1, verbose=0):

	# 3 for exact MSE, 2 for Max graph chain, 1 for graph neighbor chain
	scoring_outcome = 0

	if len(candidates) == 0:
		return -1, 0, 0

	else:

		# Find all the chains, find out which parent is the max.
		max_parent_score = 0
		max_parent_cand = ()
		graph_chains = dict()

		ns = full_neighbor_score_sweep(graph, vector_scores, n_iterations=n_graph_iterations)
		vs = vector_scores

		for candidate in candidates:
			graph_chains[candidate[0]] = find_and_score_parents(graph, vector_scores, list(), candidate[0], 0, verbose=verbose)
			if np.abs(candidate[1]) > max_parent_score:
				max_parent_score = np.abs(candidate[1])
				max_parent_cand = candidate[0]

		if max_parent_cand == attack_target:
			if verbose >= 0:
				print(f'For attack {attack_footer}: {max_parent_cand} is max-MSE')
			scoring_outcome = 3

		elif attack_target in graph_chains[max_parent_cand]:
			if verbose >= 0:
				print(f'For attack {attack_footer}: {max_parent_cand} -> {graph_chains[max_parent_cand]} is max-MSE chain')
			scoring_outcome = 2

		else:

			if attack_target in list(graph.pred[max_parent_cand].keys()):
				if verbose >= 0:
					print(f'For attack {attack_footer}: {max_parent_cand} -> {list(graph.pred[max_parent_cand].keys())} is direct parent')
				scoring_outcome = 1

			for sensor, chain in graph_chains.items():
				if attack_target in chain:
					if verbose >= 0:
						print(f'For attack {attack_footer}: {sensor} -> {chain}')
					scoring_outcome = 1
				else:
					if verbose >= 0:
						print(f'For attack {attack_footer}: {sensor} -> {chain} miss')

	vector_rank = scores_to_rank(vs, sen_to_idx(attack_target))
	neighbor_rank = scores_to_rank(ns, sen_to_idx(attack_target))

	# print(f'Scoring outcome: {scoring_outcome}')
	# print(f'Vector score ranking: {vector_rank}')
	# print(f'Neighbor score ranking: {neighbor_rank}')

	return scoring_outcome, vector_rank, neighbor_rank

def rank_attack_explanation(graph, vector_scores, attack_target, n_graph_iterations=1):

	ns = full_neighbor_score_sweep(graph, vector_scores, n_iterations=n_graph_iterations)
	vs = vector_scores
	vector_rank = scores_to_rank(vs, sen_to_idx(attack_target))
	neighbor_rank = scores_to_rank(ns, sen_to_idx(attack_target))

	return vector_rank, neighbor_rank

def aggregate_graphrank(filename, 
	graph=None, 
	suffix='in',
	selection_idx=0, 
	summary_mode=None,
	use_graph=False,
	topk=10,
	verbose=-1):

	if graph is None:
		graph = nx.readwrite.gml.read_gml('explanations-dir/graph-TEP-filtered.gml')

	graphrank_results = pickle.load(open(f'explanations-dir/{filename}', 'rb'))
	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']
	attack_patterns = ['cons', 'csum', 'line', 'lsum']
	attacks_to_iterate = get_non_pid() + get_pid() + get_xmv()

	# Options: MSE detected, MSE Graph Detected, Post MSE Detected, Post MSE Graph Detected
	# 8 attack types
	all_attack_len = len(attacks_to_iterate)
	attack_types_len = len(attack_patterns) * len(attack_types)

	all_attack_preds = dict()
	
	for an in attacks_to_iterate:
		for at in attack_types:
			for ap in attack_patterns:

				attack_footer = f'{ap}_{at}_{an}'

				if attack_footer in att_skip_list:

					if verbose >= 0:
						print(f'Skipping crashing attack {ap} {at} {an}!')
					continue

				if verbose >= 0:
					print('==================================')
					print(f'Scoring: attack {attack_footer}')
					print('==================================')

				attack_code = f'{attack_footer}_{suffix}'
				attack_quant = f'{attack_footer}_quant_{suffix}'

				if len(graphrank_results[attack_code]) > 0:

					if summary_mode == 'mean':
						selected_quant = np.abs(graphrank_results[attack_quant][:, 0:selection_idx])
						this_graph_quant = np.mean(selected_quant, axis=1)
					else:
						this_graph_quant = np.abs(graphrank_results[attack_quant][:, selection_idx])

					pred_set = top_k_predict(graph, this_graph_quant, use_graph=use_graph, topk=topk)
					all_attack_preds[attack_footer] = pred_set

				else:

					all_attack_preds[attack_footer] = None

	return all_attack_preds

def open_and_explore_graphrank(filename, 
	graph=None, 
	suffix='in', 
	types_to_iterate='all', 
	locations_to_iterate='all', 
	patterns_to_iterate='all', 
	selection_idx=0, 
	summary_mode=None,
	n_graph_iterations=1,
	verbose=0):

	supported_summaries = ['mean', 'hist']
	if summary_mode is not None and summary_mode not in supported_summaries:
		print(f'Summary mode {summary_mode} is not supported! Supported modes: {supported_summaries}')
		return None, None, None

	if graph is None:
		graph = nx.readwrite.gml.read_gml('explanations-dir/graph-TEP-filtered.gml')

	_, sensor_cols = load_train_data('TEPK', train_shuffle=True, verbose=0)
	graphrank_results = pickle.load(open(f'explanations-dir/{filename}', 'rb'))

	if types_to_iterate == 'all':
		attack_types = ['p2s', 'm2s', 'p3s', 'p5s']
	else:
		attack_types = types_to_iterate

	if patterns_to_iterate == 'all':
		attack_patterns = ['cons', 'csum', 'line', 'lsum']
	else:
		attack_patterns = patterns_to_iterate

	if locations_to_iterate == 'nonpid':
		attacks_to_iterate = get_non_pid()
	elif locations_to_iterate == 'pid':
		attacks_to_iterate = get_pid()
	elif locations_to_iterate == 'xmv':
		attacks_to_iterate = get_xmv()
	else:
		attacks_to_iterate = get_non_pid() + get_pid() + get_xmv()

	# Options: MSE detected, MSE Graph Detected, Post MSE Detected, Post MSE Graph Detected
	# 8 attack types
	all_attack_len = len(attacks_to_iterate)
	attack_types_len = len(attack_patterns) * len(attack_types)

	per_attack_scoring = np.zeros((all_attack_len, attack_types_len))
	per_attack_vr = np.zeros((all_attack_len, attack_types_len))
	per_attack_nr = np.zeros((all_attack_len, attack_types_len))

	per_attack_distance_scoring = np.zeros((all_attack_len, attack_types_len))
	per_attack_distance_candidates = np.zeros((all_attack_len, attack_types_len))

	# Some tricky indexing to build plot object
	pa_idx_x = -1
	pa_idx_y = -1

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

					per_attack_scoring[pa_idx_x, pa_idx_y] = -2
					per_attack_vr[pa_idx_x, pa_idx_y] = -1
					per_attack_nr[pa_idx_x, pa_idx_y] = -1
					continue

				if verbose >= 0:
					print('==================================')
					print(f'Scoring: attack {attack_footer}')
					print('==================================')

				attack_code = f'{attack_footer}_{suffix}'
				attack_quant = f'{attack_footer}_quant_{suffix}'

				if len(graphrank_results[attack_code]) > 0:

					# Just select from selection_idx
					if summary_mode is None:
						
						this_graph_quant = np.abs(graphrank_results[attack_quant][:, selection_idx])

						outcome, vector_rank, neighbor_rank = score_attack_explanation(graph, graphrank_results[attack_code], 
							this_graph_quant, attack_footer, an, n_graph_iterations=n_graph_iterations, verbose=verbose)

					# Take the mean from 0 - selection_idx
					elif summary_mode == 'mean':
						
						selected_quant = np.abs(graphrank_results[attack_quant][:, 0:selection_idx])
						this_graph_quant = np.mean(selected_quant, axis=1)
						
						outcome, vector_rank, neighbor_rank = score_attack_explanation(graph, graphrank_results[attack_code], 
							this_graph_quant, attack_footer, an, n_graph_iterations=n_graph_iterations, verbose=verbose)

					# Take the rolling histogram from 0 - selection_idx
					elif summary_mode == 'hist':

						# Turn into t by d
						selected_quant = np.abs(graphrank_results[attack_quant][:, 0:selection_idx]).T
						neighbor_scores = np.zeros_like(selected_quant)

						for i in range(len(selected_quant)):
							neighbor_scores[i] = full_neighbor_score_sweep(graph, selected_quant[i], 
								n_iterations=n_graph_iterations)
						
						n_sensor = selected_quant.shape[1]
						hist_score = np.zeros(n_sensor)
						graph_hist_score = np.zeros(n_sensor)

						mserank_picks = np.argmax(selected_quant, axis=1)
						graphrank_picks = np.argmax(neighbor_scores, axis=1)

						for si in range(n_sensor):
							hist_score[si] = np.sum(mserank_picks == si) 
							graph_hist_score[si] = np.sum(graphrank_picks == si) 

						attack_sensor_idx = attack_footer_to_sensor_idx(attack_footer)
						outcome = 1 # placeholder
						vector_rank = scores_to_rank(hist_score, attack_sensor_idx)
						neighbor_rank = scores_to_rank(graph_hist_score, attack_sensor_idx)

					per_attack_scoring[pa_idx_x, pa_idx_y] = outcome
					per_attack_vr[pa_idx_x, pa_idx_y] = vector_rank
					per_attack_nr[pa_idx_x, pa_idx_y] = neighbor_rank

				else:

					per_attack_scoring[pa_idx_x, pa_idx_y] = -1
					per_attack_vr[pa_idx_x, pa_idx_y] = 0
					per_attack_nr[pa_idx_x, pa_idx_y] = 0

	return per_attack_scoring, per_attack_vr, per_attack_nr

def open_and_explore_distance_elimination(filename,
	graph=None,
	suffix='in',
	locations_to_iterate='all',
	patterns_to_iterate='all',
	types_to_iterate='all',
	verbose=0,
	minimum_distance=3,
	topk_search=3,
	n_graph_iterations=1,
	use_graphscore=False):

	if graph is None:
		graph = nx.readwrite.gml.read_gml('explanations-dir/graph-TEP-filtered.gml')

	graphrank_results = pickle.load(open(f'explanations-dir/{filename}', 'rb'))

	if types_to_iterate == 'all':
		attack_types = ['p2s', 'm2s', 'p3s', 'p5s']
	else:
		attack_types = types_to_iterate

	if patterns_to_iterate == 'all':
		attack_patterns = ['cons', 'csum', 'line', 'lsum']
	else:
		attack_patterns = [patterns_to_iterate]

	if locations_to_iterate == 'nonpid':
		attacks_to_iterate = get_non_pid()
	elif locations_to_iterate == 'pid':
		attacks_to_iterate = get_pid()
	elif locations_to_iterate == 'xmv':
		attacks_to_iterate = get_xmv()
	else:
		attacks_to_iterate = get_pid() + get_xmv()

	# Options: MSE detected, MSE Graph Detected, Post MSE Detected, Post MSE Graph Detected
	# 8 attack types
	all_attack_len = len(attacks_to_iterate)
	attack_types_len = len(attack_patterns) * len(attack_types)

	per_attack_distance_scoring = np.zeros((all_attack_len, attack_types_len))
	per_attack_distance_candidates = np.zeros((all_attack_len, attack_types_len))

	# Some tricky indexing to build plot object
	pa_idx_x = -1
	pa_idx_y = -1

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

					per_attack_distance_scoring[pa_idx_x, pa_idx_y] = -1
					per_attack_distance_candidates[pa_idx_x, pa_idx_y] = -1
					continue

				if verbose >= 0:
					print('==================================')
					print(f'Scoring: attack {attack_footer}')
					print('==================================')
				attack_code = f'{attack_footer}_{suffix}'
				attack_quant = f'{attack_footer}_quant_{suffix}'

				if len(graphrank_results[attack_code]) > 0:

					# 2nd column uses standard deviation
					this_graph_quant = np.abs(graphrank_results[attack_quant][:, 0])

					if use_graphscore:
						vector_scores = full_neighbor_score_sweep(graph, this_graph_quant,
							n_iterations=n_graph_iterations)
					else:
						vector_scores = this_graph_quant

					distance_outcome, set_size = ranking_to_distance(graph, vector_scores, attack_footer, an,
						minimum_distance=minimum_distance, topk_search=topk_search, verbose=0)

					per_attack_distance_scoring[pa_idx_x, pa_idx_y] = distance_outcome
					per_attack_distance_candidates[pa_idx_x, pa_idx_y] = set_size

				else:

					per_attack_distance_scoring[pa_idx_x, pa_idx_y] = -1
					per_attack_distance_candidates[pa_idx_x, pa_idx_y] = 0

	return per_attack_distance_scoring, per_attack_distance_candidates

def generate_distance_elimination(input_config):

	pa_idx_z = 0
	max_top_k = 10
	max_top_distance = 10

	plot_obj_outcomes = np.zeros((len(input_config) * 2, max_top_k, max_top_distance))
	plot_obj_set_sizes = np.zeros((len(input_config) * 2, max_top_k, max_top_distance))

	for config in input_config:
		print('==========')
		print(f'For {config["filename"]} with {config["suffix"]}')
		print('==========')
		plot_header = config["filename"][:-4]

		for use_graphscore in [False, True]:

			for topk in range(max_top_k):
				for minimum_distance in range(max_top_distance):

					per_attack_distance_scoring, per_attack_ss = open_and_explore_distance_elimination(config['filename'],
						patterns_to_iterate='all',
						types_to_iterate='all',
						suffix=config['suffix'],
						topk_search=topk+1,
						minimum_distance=minimum_distance,
						use_graphscore=use_graphscore,
						verbose=-1)

					average_set_size_guessed = np.mean(per_attack_ss[per_attack_distance_scoring >= 0])
					correct_rate = np.sum(per_attack_distance_scoring == 1) / np.sum(per_attack_distance_scoring >= 0)

					print(f'For {plot_header} topk={topk} minimum_distance={minimum_distance}: {np.sum(per_attack_distance_scoring == 1)} / {np.sum(per_attack_distance_scoring >= 0)}')
					print(f'For {plot_header} topk={topk} minimum_distance={minimum_distance}: average set is {average_set_size_guessed}')

					plot_obj_outcomes[pa_idx_z, topk, minimum_distance] = correct_rate
					plot_obj_set_sizes[pa_idx_z, topk, minimum_distance] = average_set_size_guessed

			pa_idx_z += 1

	np.save(f'explanations-dir/graph-elimination-outcomes20.npy', plot_obj_outcomes)
	np.save(f'explanations-dir/graph-elimination-sizes20.npy', plot_obj_set_sizes)

def plot_elimination_cdf(input_config):

	graph_suffixs = ['filtered']

	for graph_suffix in graph_suffixs:

		graph = nx.readwrite.gml.read_gml(f'explanations-dir/graph-TEP-{graph_suffix}.gml')

		cdf_objs = []

		for config in input_config:

			dict_header = config["filename"][:-4]
			_, per_attack_vr, per_attack_nr = open_and_explore_graphrank(config['filename'],
					graph=graph,
					suffix=config['suffix'],
					verbose=-1)

			################

			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr >= 1], make_plot=False))
			cdf_objs.append(tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr >= 1], make_plot=False))

		labels = ['MSE TopK', 'MSE+GraphWeight TopK']
		colors = [QUAL_CMAP[0], QUAL_CMAP[4]]
		styles = ['-', 'dashed'] * 5
		mstyles = ['x']

		### Add in graph elimination results
		graphe_outcomes = np.load(f'explanations-dir/graph-elimination-outcomes20.npy')
		graphe_sizes = np.load(f'explanations-dir/graph-elimination-sizes20.npy')

		pdb.set_trace()

		fig, ax = plt.subplots(1, 1, figsize=(8, 6))

		for i in range(10):
			ax.plot(graphe_sizes[0][:,i], graphe_outcomes[0][:,i], color=QUAL_CMAP[2])
			ax.plot(graphe_sizes[1][:,i], graphe_outcomes[1][:,i], color=QUAL_CMAP[6])

		ax.scatter(graphe_sizes[0].flatten(), graphe_outcomes[0].flatten(), label='MSE Elimination', color=QUAL_CMAP[2], s=20, marker=mstyles[0])
		ax.scatter(graphe_sizes[1].flatten(), graphe_outcomes[1].flatten(), label='MSE+GraphWeight Elimination', color=QUAL_CMAP[6], s=20, marker=mstyles[0])
		
		ax.plot(cdf_objs[0][:, 0], cdf_objs[0][:, 1], label=labels[0], color=colors[0])
		ax.plot(cdf_objs[1][:, 0], cdf_objs[1][:, 1], label=labels[1], color=colors[1])

		ax.set_yticks(np.arange(0, 1.1, 0.1))
		ax.grid(which='minor', axis='y')

		ax.set_ylim([0, 1])
		ax.set_xlim([1, 53])
		ax.set_xlabel('# of Features Examined')
		ax.set_ylabel('% of attacks explained')

		ax.legend()
		fig.tight_layout()
		plt.savefig('explanations-dir/plot-cdf-scatter.pdf')
		plt.close()

def subprocesses_explain(ensemble_config, mse_config):

	graph = nx.readwrite.gml.read_gml('explanations-dir/graph-TEP-filtered.gml')
	_, sensor_cols = load_train_data('TEP', train_shuffle=True)

	scoring_results, per_attack_vr, _ = open_and_explore_subprocess_rank(ensemble_config['filename'],
		graph=graph,
		suffix=ensemble_config['suffix'],
		verbose=-1)

	print('For subprocesses MSE')
	print(f'Exactly explained: {np.sum(scoring_results == 2)} / {np.sum(scoring_results >= 0)}')
	print(f'Subprocess explained: {np.sum(scoring_results >= 1)} / {np.sum(scoring_results >= 0)}')
	print(f'average set size: {np.mean(per_attack_vr[per_attack_vr > 0])}')

	em_cdf_obj = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr > 0], make_plot=False, n_sensors=len(sensor_cols))
	auc_score = tep_plot_utils.get_auc(em_cdf_obj[:, 0], em_cdf_obj[:, 1])

	print(f'Subprocess AUC: {auc_score}')

	################

	scoring_results, per_attack_vr, per_attack_nr = open_and_explore_graphrank(
			mse_config['filename'],
			types_to_iterate='all',
			patterns_to_iterate='all',
			suffix=mse_config['suffix'],
			verbose=-1)

	print('For MSE')
	print(f'Exactly explained: {np.sum(scoring_results == 2)} / {np.sum(scoring_results >= 0)}')
	print(f'Subprocess explained: {np.sum(scoring_results >= 1)} / {np.sum(scoring_results >= 0)}')
	print(f'average set size: {np.mean(per_attack_vr[per_attack_vr > 0])}')

	mse_cdf_obj = tep_plot_utils.plot_cdf(per_attack_vr[per_attack_vr > 0], make_plot=False, n_sensors=len(sensor_cols))
	auc_score = tep_plot_utils.get_auc(mse_cdf_obj[:, 0], mse_cdf_obj[:, 1])

	print(f'MSE AUC: {auc_score}')

	mse_g_cdf_obj = tep_plot_utils.plot_cdf(per_attack_nr[per_attack_nr > 0], make_plot=False, n_sensors=len(sensor_cols))
	auc_score = tep_plot_utils.get_auc(mse_g_cdf_obj[:, 0], mse_g_cdf_obj[:, 1])

	print(f'MSE+ AUC: {auc_score}')

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	baseline_x = np.arange(0, 1.1, 0.1)
	baseline_values = np.zeros(11)
	for i in range(11):
		baseline_values[i] = len(sensor_cols) * baseline_x[i]

	ax.plot(baseline_values, baseline_x, color='black', linestyle='dashed')
	ax.plot(mse_cdf_obj[:, 0], mse_cdf_obj[:, 1], label='MSE')
	ax.plot(mse_g_cdf_obj[:, 0], mse_g_cdf_obj[:, 1], label='MSE+')
	ax.plot(em_cdf_obj[:, 0], em_cdf_obj[:, 1], label='Ensemble')

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.grid(which='minor', axis='y')

	ax.set_ylim([0, 1])
	ax.set_xlim([0, len(sensor_cols)])
	ax.set_xlabel('# of Features Examined', fontsize=16)
	ax.set_ylabel('% of Attacks Explained', fontsize=16)

	ax.legend(loc='lower right', fontsize=16)
	fig.tight_layout()

	plt.savefig('explanations-dir/plot-tep-ensemble-cdfs.pdf')
	plt.close()

	return

if __name__ == '__main__':

	# Clean up wd, set options
	import os
	os.chdir('..')

	###################################################
	### For baseline explanations
	###################################################

	input_config = [

		# { 'filename': 'gradrank-TEP-perfeat-w10-saliency_map_mse_history-tp100.pkl', 'suffix': 'in' },
		# { 'filename': 'gradrank-TEP-perfeat-w10-smooth_gradients_mse_history-tp100.pkl', 'suffix': 'in' },
		# { 'filename': 'gradrank-TEP-perfeat-w10-integrated_gradients_mse_history-tp100.pkl', 'suffix': 'in' },
		# { 'filename': 'gradrank-TEP-perfeat-w10-expected_gradients_mse_history-tp100.pkl', 'suffix': 'in' },

		# { 'filename': 'limerank-TEP-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		# { 'filename': 'shaprank-TEP-w10-perfeat-summarized.pkl', 'suffix': 'in' },
		
		# { 'filename': 'cfrank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		# { 'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' },

	]

	subprocesses_explain(
		{'filename': 'ensemble-mserank-CNN-TEP-l2-hist50-kern3-units64-w10-perfeat-tp100.pkl', 'suffix': 'in' },
		{'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' }
		)

	generate_distance_elimination([{'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' }])
	plot_elimination_cdf([{'filename': 'mserank-TEP-w10-perfeat-tp100.pkl', 'suffix': 'in' }])
