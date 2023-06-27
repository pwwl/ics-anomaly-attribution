import numpy as np
import matplotlib.pyplot as plt

import pdb
import pickle

from data_loader import load_train_data, load_test_data

import attack_utils

np.set_printoptions(suppress=True)

HOUR = 2000

def ndss_paper_example():

	all_df = []

	## COLLECT SWAT RANKINGS
	dataset = 'SWAT'
	Xtest, Ytest, sensor_cols = load_test_data(dataset, no_transform=True)
	
	attacks, labels = attack_utils.get_attack_indices(dataset)
	sds = attack_utils.get_attack_sds(dataset)
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	lookup_name = f'GRU-SWAT-l2-hist50-units64-results_ns1'
	validation_errors = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')
	test_errors = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
	
	validation_instance_errors = np.mean(validation_errors, axis=1)
	test_instance_errors = np.mean(test_errors, axis=1)

	cutoff = np.quantile(validation_instance_errors, 0.9995)

	# Plot the raw feature
	example_attack = 10
	col_idx = 37
	attack_start = attacks[example_attack][0]
	attack_end = attacks[example_attack][-1]
	attack_len = attack_end - attack_start
	
	plot_start = attack_start - 100
	plot_end = plot_start + 400
	
	fig, ax = plt.subplots(5, 1, figsize=(14,10))

	raw_range = np.arange(plot_start, plot_end)
	ax[0].plot(Xtest[raw_range,col_idx], lw=2)
	#ax[0].fill_between(np.arange(100, attack_len+100), 11, 19, facecolor='#ff7777')
	#ax[0].text(2 + 100, 16.5, 'anomaly\nstart', fontsize=14)
	#ax[0].text(2 + 100 + attack_len, 16.5, 'anomaly\nend', fontsize=14)
	ax[0].vlines([100, 100+attack_len], ymin=0, ymax=19, color='grey', linestyles='--')
	ax[0].fill_between(np.arange(plot_end - plot_start), 11, Xtest[raw_range,col_idx])
	ax[0].set_ylim([11, 19])

	# Plot the MSEs
	col_idx = 37
	data_shift_amt = 51
	instance_range = test_instance_errors[plot_start-data_shift_amt : plot_end-data_shift_amt]
	column_instance_range = test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx]

	ax[1].plot(test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx], lw=2)
	ax[1].fill_between(np.arange(plot_end - plot_start), 0, test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx])

	ax[1].vlines([100, 100+attack_len], ymin=0, ymax=np.max(column_instance_range), color='grey', linestyles='--')
	ax[2].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[3].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[4].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')

	first_det = detection_points[lookup_name][example_attack] + data_shift_amt + 50
	ideal1 = 100
	ideal2 = ideal1 + 50
	
	ax[2].fill_between(np.arange(ideal1-50, ideal1+1), 0, np.max(instance_range), facecolor='#bbbbbb')
	ax[2].plot(instance_range, lw=2)
	ax[2].fill_between(np.arange(plot_end - plot_start), 0, instance_range)

	ax[3].fill_between(np.arange(ideal2-50, ideal2+1), 0, np.max(instance_range), facecolor='#bbbbbb')
	ax[3].plot(instance_range, lw=2)
	ax[3].fill_between(np.arange(plot_end - plot_start), 0, instance_range)

	ax[4].fill_between(np.arange(first_det-50, first_det+1), 0, np.max(instance_range), facecolor='#bbbbbb')
	ax[4].plot(instance_range, lw=2)
	ax[4].fill_between(np.arange(plot_end - plot_start), 0, instance_range)

	ax[2].scatter([ideal1], [instance_range[ideal1]], color='red', zorder=3)
	#ax[2].vlines([ideal1], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[3].scatter([ideal2], [instance_range[ideal2]], color='red', zorder=3)
	#ax[3].vlines([ideal2], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[4].scatter([first_det], [instance_range[first_det]], color='red', zorder=3)
	#ax[4].vlines([first_det], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')

	#ax[2].text(ideal1-50, 5.0, 'model\ninput', fontsize=14)
	#ax[2].text(ideal1 + 3, 5.0, 'attack\nstart', fontsize=14)
	#ax[2].text(first_det + 3, 5.0, 'first\ndetection', fontsize=14)

	# ax[0].set_title('AIT504 raw sensor value', fontsize=20)
	# ax[1].set_title('AIT504 prediction error', fontsize=20)
	# ax[2].set_title('MSE (overall) - model input begins at attack start', fontsize=20)
	# ax[3].set_title('MSE (overall) - model input ends at attack start', fontsize=20)
	# ax[4].set_title('MSE (overall) - model input begins at detection point', fontsize=20)

	#ax[0].set_ylabel('spacing', fontsize=20, rotation='horizontal', color='white')
	# ax[1].set_ylabel('AIT504 prediction error', fontsize=20, rotation='horizontal')
	# ax[2].set_ylabel('MSE (overall)\n model input\nbegins at attack start', fontsize=20, rotation='horizontal')
	# ax[3].set_ylabel('MSE (overall)\n model input\nends at attack start', fontsize=20, rotation='horizontal')
	# ax[4].set_ylabel('MSE (overall)\n model input\nbegins at detection point', fontsize=20, rotation='horizontal')
	
	ax[0].set_xticks([])
	ax[1].set_xticks([])
	ax[2].set_xticks([])
	ax[3].set_xticks([])

	# ax[0].set_ylabel('Value', fontsize=16)
	# ax[1].set_ylabel('Error', fontsize=16)
	# ax[2].set_ylabel('Error', fontsize=16)
	# ax[3].set_ylabel('Error', fontsize=16)
	# ax[4].set_ylabel('Error', fontsize=16)
	ax[4].set_xlabel('Time (s)', fontsize=16)

	fig.tight_layout(rect=[0.15, 0, 1, 1])

	plt.savefig('plot-ndss-timing-example.png')
	plt.close()

	#pdb.set_trace()

	print('All done!')

def ndss_paper_example_4row():

	all_df = []

	## COLLECT SWAT RANKINGS
	dataset = 'SWAT'
	Xtest, Ytest, sensor_cols = load_test_data(dataset, no_transform=True)
	
	attacks, labels = attack_utils.get_attack_indices(dataset)
	sds = attack_utils.get_attack_sds(dataset)
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	lookup_name = f'GRU-SWAT-l2-hist50-units64-results_ns1'
	validation_errors = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')
	test_errors = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
	
	validation_instance_errors = np.mean(validation_errors, axis=1)
	test_instance_errors = np.mean(test_errors, axis=1)

	cutoff = np.quantile(validation_instance_errors, 0.9995)

	# Plot the raw feature
	example_attack = 10
	col_idx = 37
	attack_start = attacks[example_attack][0]
	attack_end = attacks[example_attack][-1]
	attack_len = attack_end - attack_start
	
	plot_start = attack_start - 100
	plot_end = plot_start + 400
	
	fig, ax = plt.subplots(4, 1, figsize=(14,8))

	raw_range = np.arange(plot_start, plot_end)
	ax[0].plot(Xtest[raw_range,col_idx], lw=2)
	#ax[0].fill_between(np.arange(100, attack_len+100), 11, 19, facecolor='#ff7777')
	#ax[0].text(2 + 100, 16.5, 'anomaly\nstart', fontsize=14)
	#ax[0].text(2 + 100 + attack_len, 16.5, 'anomaly\nend', fontsize=14)
	ax[0].vlines([100, 100+attack_len], ymin=0, ymax=19, color='grey', linestyles='--')
	ax[0].fill_between(np.arange(plot_end - plot_start), 11, Xtest[raw_range,col_idx])
	ax[0].set_ylim([11, 19])

	# Plot the MSEs
	col_idx = 37
	data_shift_amt = 51
	instance_range = test_instance_errors[plot_start-data_shift_amt : plot_end-data_shift_amt]
	column_instance_range = test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx]

	ax[1].plot(test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx], lw=2)
	ax[1].fill_between(np.arange(plot_end - plot_start), 0, test_errors[plot_start-data_shift_amt : plot_end-data_shift_amt, col_idx])

	ax[1].vlines([100, 100+attack_len], ymin=0, ymax=np.max(column_instance_range), color='grey', linestyles='--')
	#ax[2].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[2].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')
	ax[3].vlines([100, 100+attack_len], ymin=0, ymax=np.max(instance_range), color='grey', linestyles='--')

	first_det = detection_points[lookup_name][example_attack] + data_shift_amt + 50
	ideal1 = 100
	ideal2 = ideal1 + 50

	ax[2].plot(instance_range, lw=2)
	ax[2].fill_between(np.arange(plot_end - plot_start), 0, instance_range)
	# ax[2].fill_between(np.arange(ideal2-50, ideal2+1), 0, np.max(instance_range), facecolor='#bbbbbb')
	# ax[2].scatter([ideal2], [instance_range[ideal2]], color='red', zorder=3)

	ax[3].fill_between(np.arange(first_det-50, first_det+1), 0, np.max(instance_range), facecolor='#bbbbbb')
	ax[3].plot(instance_range, lw=2)
	ax[3].fill_between(np.arange(plot_end - plot_start), 0, instance_range)
	ax[3].scatter([first_det], [instance_range[first_det]], color='red', zorder=3)
	
	ax[0].set_xticks([])
	ax[1].set_xticks([])
	ax[2].set_xticks([])
	
	ax[0].set_yticks([])
	ax[1].set_yticks([])
	ax[2].set_yticks([])
	ax[3].set_yticks([])

	ax[3].tick_params(axis='x', which='major', labelsize=18)
	ax[3].set_xlabel('Time (s)', fontsize=24)

	fig.tight_layout(rect=[0.15, 0, 1, 1])

	plt.savefig('plot-ndss-timing-example-4.png')
	plt.close()

	#pdb.set_trace()

	print('All done!')

def ccs_paper_example():

	all_df = []

	## COLLECT SWAT RANKINGS
	dataset = 'SWAT'
	Xtest, Ytest, sensor_cols = load_test_data(dataset, no_transform=True)
	
	attacks, labels = attack_utils.get_attack_indices(dataset)
	sds = attack_utils.get_attack_sds(dataset)
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	lookup_name = f'GRU-SWAT-l2-hist50-units64-results_ns1'
	validation_errors = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')
	test_errors = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
	
	validation_instance_errors = np.mean(validation_errors, axis=1)
	test_instance_errors = np.mean(test_errors, axis=1)

	cutoff = np.quantile(validation_instance_errors, 0.9995)

	# Plot the raw feature
	example_attack = 10
	col_idx = 37
	attack_start = attacks[example_attack][0]
	
	plot_start = attack_start - 50
	plot_end = plot_start + 200
	
	fig, ax = plt.subplots(3, 1, figsize=(10,5))

	raw_range = np.arange(plot_start, plot_end)
	ax[0].plot(Xtest[raw_range,col_idx], lw=2)
	ax[0].fill_between(np.arange(200), 11, Xtest[raw_range,col_idx])
	ax[0].set_ylim([11, 17])

	# Plot the MSEs
	col_idx = 37
	shift_amt = 51
	instance_range = test_instance_errors[plot_start-shift_amt : plot_end-shift_amt]

	ax[1].plot(test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx], lw=2)
	ax[1].fill_between(np.arange(200), 0, test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx])

	ax[2].plot(instance_range, lw=2)
	ax[2].fill_between(np.arange(200), 0, instance_range)

	first_det = detection_points[lookup_name][example_attack] + shift_amt
	ideal1 = shift_amt
	ideal2 = 50 + shift_amt 
	#ax[2].vlines([ideal_det, first_det], ymin=0, ymax=3, color='green', linestyles='--')
	ax[2].scatter([ideal1, ideal2, first_det], [instance_range[ideal1], instance_range[ideal2], instance_range[first_det]], color='green')
	ax[2].text(ideal1, instance_range[ideal1] + 0.5, 'ideal1', fontsize=16)
	ax[2].text(ideal2 - 3, instance_range[ideal2] + 0.5, 'ideal2', fontsize=16)
	ax[2].text(first_det - 12, instance_range[first_det] - 1, 'real', fontsize=16)
	ax[2].hlines(cutoff, xmin=0, xmax=200, color='red', linestyles='--')

	ax[0].set_title('AIT504 raw sensor value', fontsize=20)
	ax[1].set_title('AIT504 prediction error', fontsize=20)
	ax[2].set_title('MSE', fontsize=20)
	
	ax[0].set_xticks([])
	ax[1].set_xticks([])

	ax[0].set_ylabel('value', fontsize=16)
	ax[1].set_ylabel('error', fontsize=16)
	ax[2].set_ylabel('error', fontsize=16)

	fig.tight_layout()

	plt.savefig('plot-ccs-timing-example.pdf')
	plt.close()

	print('All done!')

def slide_example():

	all_df = []

	## COLLECT SWAT RANKINGS
	dataset = 'SWAT'
	Xtest, Ytest, sensor_cols = load_test_data(dataset, no_transform=True)
	
	attacks, labels = attack_utils.get_attack_indices(dataset)
	sds = attack_utils.get_attack_sds(dataset)
	detection_points = pickle.load(open('ccs-storage/detection-points.pkl', 'rb'))

	lookup_name = f'GRU-SWAT-l2-hist50-units64-results_ns1'
	validation_errors = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')
	test_errors = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
	
	validation_instance_errors = np.mean(validation_errors, axis=1)
	test_instance_errors = np.mean(test_errors, axis=1)

	cutoff = np.quantile(validation_instance_errors, 0.9995)

	# Plot the raw feature
	example_attack = 10
	col_idx = 37
	attack_start = attacks[example_attack][0]
	
	plot_start = attack_start - 50
	plot_end = plot_start + 200
	
	fig, ax = plt.subplots(2, 1, figsize=(10,5))

	raw_range = np.arange(plot_start, plot_end)
	ax[0].plot(Xtest[raw_range,col_idx], lw=2)
	ax[0].fill_between(np.arange(200), 11, Xtest[raw_range,col_idx])
	ax[0].set_ylim([11, 17])

	# Plot the MSEs
	col_idx = 37
	shift_amt = 51
	instance_range = test_instance_errors[plot_start-shift_amt : plot_end-shift_amt]

	# ax[1].plot(test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx], lw=2)
	# ax[1].fill_between(np.arange(200), 0, test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx])

	ax[1].plot(instance_range, lw=2)
	ax[1].fill_between(np.arange(200), 0, instance_range)

	first_det = detection_points[lookup_name][example_attack] + shift_amt
	ideal1 = shift_amt
	ideal2 = 50 + shift_amt 
	ax[1].scatter([ideal1, first_det], [instance_range[ideal1], instance_range[first_det]], color='green')
	ax[1].text(ideal1, instance_range[ideal1] + 0.5, 'attack start', fontsize=16)
	#ax[1].text(ideal2 - 3, instance_range[ideal2] + 0.5, 'ideal2', fontsize=16)
	ax[1].text(first_det + 1, instance_range[first_det] - 1, 'detection', fontsize=16)
	ax[1].hlines(cutoff, xmin=0, xmax=200, color='red', linestyles='--')

	ax[0].set_title('AIT504 raw sensor value', fontsize=20)
	ax[1].set_title('MSE', fontsize=20)
	
	ax[0].set_xticks([])
	#ax[1].set_xticks([])

	ax[0].set_ylabel('value', fontsize=16)
	#ax[1].set_ylabel('error', fontsize=16)
	ax[1].set_ylabel('error', fontsize=16)

	fig.tight_layout()

	plt.savefig('plot-mse-example.pdf')
	plt.close()

	print('All done!')

def all_examples():

	all_df = []

	## COLLECT SWAT RANKINGS
	dataset = 'SWAT'
	models = ['CNN', 'GRU', 'LSTM']
	
	Xtest, Ytest, sensor_cols = load_test_data(dataset, no_transform=True)
	attacks, labels = attack_utils.get_attack_indices(dataset)
	sds = attack_utils.get_attack_sds(dataset)

	for mm in models:

		if mm == 'CNN':
			lookup_name = f'{mm}-SWAT-l2-hist50-kern3-units64-results_ns1'
		else:
			lookup_name = f'{mm}-SWAT-l2-hist50-units64-results_ns1'

		validation_errors = np.load(f'ccs-storage/mses-val-{lookup_name}-{dataset}-ns.npy')
		test_errors = np.load(f'ccs-storage/mses-{lookup_name}-{dataset}-ns.npy')
		test_errors_corrected = test_errors.copy()
		test_errors_corrected[:,5] = 0

		validation_instance_errors = np.mean(validation_errors, axis=1)
		test_instance_errors = np.mean(test_errors, axis=1)
		test_instance_errors_corrected = np.mean(test_errors_corrected, axis=1)

		cutoff = np.quantile(validation_instance_errors, 0.9995)

		# Plot the raw feature
		for atk_obj in sds:

			atk_idx = atk_obj[0]
			col_name = atk_obj[1]
			col_idx = sensor_cols.index(col_name)
			attack_start = attacks[atk_idx][0]
			
			plot_start = attack_start - 50
			plot_end = plot_start + 200
			
			fig, ax = plt.subplots(3, 1, figsize=(10,5))

			raw_range = np.arange(plot_start, plot_end)
			ax[0].plot(Xtest[raw_range,col_idx], lw=2)
			ax[0].fill_between(np.arange(200), 0, Xtest[raw_range,col_idx])
			#ax[0].set_ylim([11, 17])

			# Plot the MSEs
			shift_amt = 51
			instance_range = test_instance_errors_corrected[plot_start-shift_amt : plot_end-shift_amt]

			ax[1].plot(test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx], lw=2)
			ax[1].fill_between(np.arange(200), 0, test_errors[plot_start-shift_amt : plot_end-shift_amt, col_idx])

			ax[2].plot(instance_range, lw=2)
			ax[2].fill_between(np.arange(200), 0, instance_range)

			#first_det = detection_points[lookup_name][atk_idx] + shift_amt
			ideal1 = shift_amt
			ideal2 = 50 + shift_amt 
			#ax[2].vlines([ideal_det, first_det], ymin=0, ymax=3, color='green', linestyles='--')
			#ax[2].scatter([ideal1, ideal2, first_det], [instance_range[ideal1], instance_range[ideal2], instance_range[first_det]], color='green')
			#ax[2].text(ideal1, instance_range[ideal1] + 0.5, 'ideal1', fontsize=16)
			#ax[2].text(ideal2 - 3, instance_range[ideal2] + 0.5, 'ideal2', fontsize=16)
			#ax[2].text(first_det - 12, instance_range[first_det] - 1, 'real', fontsize=16)
			ax[2].hlines(cutoff, xmin=0, xmax=200, color='red', linestyles='--')

			ax[0].set_title(f'{col_name} raw sensor value', fontsize=20)
			ax[1].set_title(f'{col_name} prediction error', fontsize=20)
			ax[2].set_title('MSE', fontsize=20)
			
			ax[0].set_xticks([])
			ax[1].set_xticks([])

			ax[0].set_ylabel('value', fontsize=16)
			ax[1].set_ylabel('error', fontsize=16)
			ax[2].set_ylabel('error', fontsize=16)

			fig.tight_layout()

			plt.savefig(f'plot-{mm}-{atk_idx}-{col_name}-example.png')
			plt.close()

			print(f'Done {mm} {atk_idx} {col_name}')

if __name__ == "__main__":
	
	ndss_paper_example()
	ndss_paper_example_4row()
