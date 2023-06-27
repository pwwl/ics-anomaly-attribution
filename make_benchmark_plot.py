
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib
import scipy.stats as ss

import tep_plot_utils

matplotlib.rcParams['pdf.fonttype'] = 42

def plot_benchmark():

	datasets = ['SWAT', 'WADI', 'TEP']
	models = ['CNN', 'GRU', 'LSTM']
	colors = ['#000000', '#FD8046', '#D36C50', '#AA585B', '#804565', '#99E6B3', '#73CDA7', '#4DB39C', '#269A90', '#008083']
	methods = ['SM', 'SG', 'IG', 'EG', 'CF-Add', 'CF-Sub', 'LIME', 'SHAP', 'LEMNA']
	ncols = [51, 119, 53]
	run_name = 'results_ns1'

	fig, ax = plt.subplots(1, 3, figsize=(14, 4))
	width = 0.075

	for dx in range(len(datasets)):

		dataset = datasets[dx] 
		plot_obj = np.zeros((len(models), len(methods) + 1))

		for mx in range(len(models)):
			
			model = models[mx]

			if model == 'CNN':
				lookup_name = f'{model}-{dataset}-l2-hist50-kern3-units64-{run_name}'
			else:
				lookup_name = f'{model}-{dataset}-l2-hist50-units64-{run_name}'

			for ex in range(len(methods)):

				method = methods[ex]
				scores = np.load(f'meta-storage/model-benchmark/benchmark-{lookup_name}-{method}.npy')

				# MSE rank vs EXP rank
				plot_obj[mx, 0] = np.mean(scores[0])
				plot_obj[mx, ex + 1] = np.mean(scores[2])

				# MSE ndcg vs EXP ndcg
				# plot_obj[mx, 0] = np.mean(scores[4])
				# plot_obj[mx, ex + 1] = np.mean(scores[5])

		if dx == 0:
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 9/2 * width, plot_obj[:,0], width, label='MSE', color=colors[0])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 7/2 * width, plot_obj[:,1], width, label='SM', color=colors[1])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 5/2 * width, plot_obj[:,2], width, label='SG', color=colors[2])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 3/2 * width, plot_obj[:,3], width, label='IG', color=colors[3])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 1/2 * width, plot_obj[:,4], width, label='EG', color=colors[4])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 1/2 * width, plot_obj[:,5], width, label='CF-Add', color=colors[5])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 3/2 * width, plot_obj[:,6], width, label='CF-Sub', color=colors[6])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 5/2 * width, plot_obj[:,7], width, label='LIME', color=colors[7])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 7/2 * width, plot_obj[:,8], width, label='SHAP', color=colors[8])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 9/2 * width, plot_obj[:,9], width, label='LEMNA', color=colors[9])
		else:
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 9/2 * width, plot_obj[:,0], width,color=colors[0])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 7/2 * width, plot_obj[:,1], width,color=colors[1])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 5/2 * width, plot_obj[:,2], width,color=colors[2])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 3/2 * width, plot_obj[:,3], width,color=colors[3])
			ax[dx].bar(np.arange(plot_obj.shape[0]) - 1/2 * width, plot_obj[:,4], width,color=colors[4])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 1/2 * width, plot_obj[:,5], width,color=colors[5])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 3/2 * width, plot_obj[:,6], width,color=colors[6])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 5/2 * width, plot_obj[:,7], width,color=colors[7])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 7/2 * width, plot_obj[:,8], width,color=colors[8])
			ax[dx].bar(np.arange(plot_obj.shape[0]) + 9/2 * width, plot_obj[:,9], width,color=colors[9])

		ax[dx].set_ylim([0, np.max(plot_obj) + 5])
		#ax[dx].set_ylim([0, 1])
		ax[dx].set_xticks(np.arange(3))
		ax[dx].set_xticklabels(models, fontsize=16)
		ax[dx].set_title(datasets[dx], fontsize=20)

		print(f'{models[0]} {dataset} {plot_obj[0]}')
		print(f'{models[1]} {dataset} {plot_obj[1]}')
		print(f'{models[2]} {dataset} {plot_obj[2]}')

	#ax[0].set_ylabel('Average nDCG', fontsize=14)
	ax[0].set_ylabel('Perturbed Feature AvgRank', fontsize=14)

	fig.legend(bbox_to_anchor=(0.5, 1), ncol=10, loc='upper center', fontsize=14)
	fig.tight_layout(rect=[0, 0, 1, 0.9])

	plt.savefig('plot-benchmark.pdf')

if __name__ == '__main__':
	
	plot_benchmark()
	