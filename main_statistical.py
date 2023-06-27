import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

import pdb
import pickle

import sys
sys.path.append('explain-eval-manipulations')

from data_loader import load_train_data, load_test_data
from main_train import load_saved_model
from tep_utils import load_tep_attack, attack_footer_to_sensor_idx, scores_to_rank, idx_to_sen, sen_to_idx
from tep_utils import get_pid, get_non_pid, get_xmv, get_footer_list

import attack_utils

np.set_printoptions(suppress=True)

HOUR = 2000

def process_ar():

	all_df = []

	## COLLECT SWAT RANKINGS
	datasets = ['SWAT', 'WADI']
	lag = 10

	for dataset in datasets:
		
		Xtest, Ytest, sensor_cols = load_test_data(dataset)
		attacks, labels = attack_utils.get_attack_indices(dataset)
		sds = attack_utils.get_attack_sds(dataset)

		ar_full = np.loadtxt(f'ccs-storage/AR-scores-{dataset}.csv', delimiter=',')

		real_rankings = list()
		lookup_names = list()

		for sd_idx in range(len(sds)):
		
			sd_obj = sds[sd_idx]
			atk_idx = sd_obj[0]
			label = sd_obj[1]
			col_idx = sensor_cols.index(label)

			attack_idxs = attacks[atk_idx]
			att_start = attack_idxs[0] - lag - 1
			
			if att_start < 0:
				continue

			ar_scores = ar_full[att_start:att_start+100, :]
			score_slice = ar_scores[lag]
			score_slice[np.isnan(score_slice)] = 0
			ranking = scores_to_rank(score_slice, col_idx)
			
			real_rankings.append(ranking)
			lookup_names.append(f'{dataset}_{sd_idx}')

		df = pd.DataFrame({
			'attack': lookup_names,
			'ar_rank': real_rankings
		})

		pickle.dump(df, open(f'ccs-storage/idealdet-AR-{dataset}.pkl', 'wb'))
		pdb.set_trace()

	## COLLECT TEP RANKINGS
	attack_footers = get_footer_list(patterns=['cons'])
	ar_rankings = list()

	for footer in attack_footers:

		#### PASAD Scoring
		ar_full = np.loadtxt(f'ccs-storage/AR-scores-TEP-{footer}.csv', delimiter=',')
		ar_scores = ar_full[9990:13990, :]
	
		ar_slice = ar_scores[lag]
		ar_slice[np.isnan(ar_slice)] = 0

		splits = footer.split("_")
		pattern = splits[0]
		mag = splits[1]
		label = splits[2]
		col_idx = sen_to_idx(label)
		ranking = scores_to_rank(ar_slice, col_idx)
		ar_rankings.append(ranking)

		print(f'for attack {footer}: ranking {ranking}')

	df = pd.DataFrame({
		'attack': attack_footers,
		'ar_rank': ar_rankings
	})

	pickle.dump(df, open(f'ccs-storage/idealdet-AR-TEP.pkl', 'wb'))
	pdb.set_trace()

	print('All done!')

def process_ar_detect():

	all_df = []
	detection_lookup = pickle.load(open('ccs-storage/detection-points-AR.pkl', 'rb'))

	## COLLECT SWAT RANKINGS
	datasets = ['SWAT', 'WADI']
	lag = 10

	for dataset in datasets:
		
		Xtest, Ytest, sensor_cols = load_test_data(dataset)
		attacks, labels = attack_utils.get_attack_indices(dataset)
		sds = attack_utils.get_attack_sds(dataset)

		ar_full = np.loadtxt(f'ccs-storage/AR-scores-{dataset}.csv', delimiter=',')

		detect_points = list()
		real_rankings = list()
		lookup_names = list()
		model_lookup = detection_lookup[f'AR-{dataset}']

		for sd_idx in range(len(sds)):
		
			sd_obj = sds[sd_idx]
			atk_idx = sd_obj[0]
			label = sd_obj[1]
			col_idx = sensor_cols.index(label)

			attack_idxs = attacks[atk_idx]
			att_start = attack_idxs[0] - lag - 1
			
			if att_start < 0:
				continue

			if atk_idx not in model_lookup:
				print(f'Attack {atk_idx} missed.')
				continue

			detect_idx = model_lookup[atk_idx]
			score_slice = ar_full[att_start+detect_idx]
			ranking = scores_to_rank(score_slice, col_idx)
			
			detect_points.append(detect_idx)
			real_rankings.append(ranking)
			lookup_names.append(f'{dataset}_{sd_idx}')

			print(f'For {dataset} atk {atk_idx}: ranking {ranking}')

		df = pd.DataFrame({
			'attack': lookup_names,
			'ar_rank': real_rankings,
			'detect_point': detect_points
		})

		pickle.dump(df, open(f'ccs-storage/realdet-AR-{dataset}.pkl', 'wb'))

	## COLLECT TEP RANKINGS
	attack_footers = get_footer_list(patterns=['cons'])
	ar_rankings = list()
	ar_detect_points = list()
	detected_footers = list()
	model_lookup = detection_lookup[f'AR-TEP']

	for footer in attack_footers:

		#### PASAD Scoring
		ar_full = np.loadtxt(f'ccs-storage/AR-scores-TEP-{footer}.csv', delimiter=',')
		ar_scores = ar_full[9990:13990, :]
	
		if footer not in model_lookup:
			print(f'Attack {footer} missed.')
			continue

		detect_idx = model_lookup[footer]

		ar_slice = ar_scores[detect_idx]

		splits = footer.split("_")
		label = splits[2]
		col_idx = sen_to_idx(label)
		
		ranking = scores_to_rank(ar_slice, col_idx)
		ar_detect_points.append(detect_idx)
		ar_rankings.append(ranking)
		detected_footers.append(footer)

		print(f'for attack {footer}: ranking {ranking}')

	df = pd.DataFrame({
		'attack': detected_footers,
		'ar_rank': ar_rankings,
		'detect_point': ar_detect_points
	})

	pickle.dump(df, open(f'ccs-storage/realdet-AR-TEP.pkl', 'wb'))
	pdb.set_trace()

	print('All done!')

def process_pasad():

	all_df = []

	## COLLECT SWAT RANKINGS
	datasets = ['SWAT', 'WADI']
	lag = 5000
	select_points = {'SWAT': 300, 'WADI': 500, 'TEP': 4000}

	for dataset in datasets:
		
		Xtest, Ytest, sensor_cols = load_test_data(dataset)
		attacks, labels = attack_utils.get_attack_indices(dataset)
		sds = attack_utils.get_attack_sds(dataset)
		select_point = select_points[dataset]

		pasad = pd.read_csv(f'/git/pasad/Matlab code/pasad_departure_scores_{dataset}.csv', header=None)
	
		real_rankings = list()
		lookup_names = list()

		for sd_idx in range(len(sds)):
		
			sd_obj = sds[sd_idx]
			atk_idx = sd_obj[0]
			label = sd_obj[1]
			col_idx = sensor_cols.index(label)

			attack_idxs = attacks[atk_idx]
			att_start = attack_idxs[0] - lag
			
			if atk_idx > 1:	
				print(attacks[atk_idx][0] - attacks[atk_idx-1][0])

			if att_start < 0:
				continue

			pasad_slice = pasad.iloc[att_start:att_start+10000, :].values
			pasad_scores = pasad_slice[select_point]
			pasad_scores[np.isnan(pasad_scores)] = 0
			ranking = scores_to_rank(pasad_scores, col_idx)
			
			real_rankings.append(ranking)
			lookup_names.append(f'{dataset}_{sd_idx}')

		df = pd.DataFrame({
			'attack': lookup_names,
			'pasad_rank': real_rankings
		})

		pickle.dump(df, open(f'idealdet-PASAD-{dataset}.pkl', 'wb'))

	## COLLECT TEP RANKINGS
	attack_footers = get_footer_list(patterns=['cons'])
	tep_rankings = list()
	select_point = select_points['TEP']

	for footer in attack_footers:

		#### PASAD Scoring
		pasad = pd.read_csv(f'/home/clementf/git/cylab/pasad/Matlab code/pasad_departure_scores_{footer}.csv', header=None)
		
		# For TEP, lag was 5000. So the attack from 10000-14000 should be from 5000-9000
		pasad_slice = pasad.iloc[5000:10000, :53].values
	
		pasad_scores = pasad_slice[select_point]
		pasad_scores[np.isnan(pasad_scores)] = 0

		splits = footer.split("_")
		pattern = splits[0]
		mag = splits[1]
		label = splits[2]
		col_idx = sen_to_idx(label)
		ranking = scores_to_rank(pasad_scores, col_idx)
		tep_rankings.append(ranking)

		print(f'for attack {footer}: ranking {ranking}')

	df = pd.DataFrame({
		'attack': attack_footers,
		'pasad_rank': tep_rankings
	})

	pickle.dump(df, open(f'idealdet-PASAD-TEP.pkl', 'wb'))

	print('All done!')

if __name__ == "__main__":

	process_ar()
	process_ar_detect()
