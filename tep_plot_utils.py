import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import metrics

from sklearn import metrics as skmetrics
import pdb

def get_xmv():

	xmv_attack_numbers = ['a1', 'a2', 'a3', 'a4', 'a6', 'a7', 'a8', 'a10', 'a11']
	return xmv_attack_numbers

def get_non_pid():

	non_pid_sensors = ['s5', 's6', 's13', 's16', 's18', 's19', 's20']
	return non_pid_sensors

def get_pid():

	pid_sensors = ['s1', 's2', 's3', 's4', 's7', 's8', 's9', 's10', 's11', 's12', 's14', 's15', 's17', 's23', 's25', 's40']
	return pid_sensors

def get_attack_ticklabels():

	attack_patterns = ['cons', 'csum', 'line', 'lsum']
	attack_types = ['p2s', 'm2s', 'p3s', 'p5s']

	yticklabels = []
	for at in attack_types:
		for ap in attack_patterns:
			if ap == 'cons':
				yticklabels.append(f'cnst_{at}')
			else:
				yticklabels.append(f'{ap}_{at}')

	return yticklabels

def plot_compare(Yhat, Ytrue, fname='compare.png'):

	f1_score = metrics.f1_score(Yhat, Ytrue)

	fig, ax = plt.subplots(1, 1, figsize=(10, 5))

	ax.set_title(f'Case #1: F1 = {f1_score:.3f}', fontsize=42)
	ax.plot(Ytrue, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
	ax.fill_between(np.arange(len(Ytrue)), 0, Ytrue, color = 'lightcoral')
	ax.plot(-1 * Yhat, color = '0.25', label = 'Predicted')
	ax.fill_between(np.arange(len(Yhat)), -1 * Yhat, 0, color = '0.25')

	ax.set_yticks([-1, 1])
	ax.set_yticklabels(['Prediction', 'Attack'], fontsize=36)
	ax.set_xlabel('Time', fontsize=36)
	ax.tick_params(axis='x', labelsize=24)

	fig.tight_layout()
	plt.savefig(fname)
	plt.close()

def plot_cdf(ranking_list, make_plot=True, fname='cdf.png', n_sensors = 53):

	# Store: for each topk value, what proportion of attacks are captured
	cdf_plot_obj = np.zeros((n_sensors + 1, 2))
	cdf_plot_obj[0, 0] = 0
	cdf_plot_obj[0, 1] = 0

	for i in range(n_sensors):
		cdf_plot_obj[i+1, 0] = i+1
		cdf_plot_obj[i+1, 1] = np.mean(ranking_list <= i+1)

	if make_plot:
		fig, ax = plt.subplots(1, 1, figsize=(8, 8))
		ax.plot(cdf_plot_obj[:, 0], cdf_plot_obj[:, 1])
		ax.set_ylim([0, 1])

		fig.tight_layout()
		plt.savefig(fname)
		plt.close()

	return cdf_plot_obj

def get_time(ranking_list, make_plot=False, fname='time.png', n_sensors = 53):

	# Amount of time spent searching
	time_per_sensor = 1
	return time_per_sensor * np.sum(ranking_list - 1) / len(ranking_list)

def get_time_improvement(ranking_list, make_plot=False, fname='time.png', n_sensors = 53):

	# Amount of time spent searching
	time_per_sensor = 1
	avg_time = time_per_sensor * np.sum(ranking_list - 1) / len(ranking_list)

	random_time = (n_sensors - 1) / 2

	return (random_time - avg_time) / random_time

def get_auc(cdf_x, cdf_y):
	return skmetrics.auc(cdf_x / np.max(cdf_x), cdf_y)
