import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pdb
import pickle
from utils import metrics

# FEATURE_SETS = [
# 	['s6', 's7', 's8', 's9', 's21', 's23', 's24', 's25', 's26', 's27', 's28', 'a10'],
# 	['s15', 's16', 's17', 's18', 's19', 'a8'],
# 	['s11', 's12', 's13', 's14', 's22', 'a7'],
# ]

def get_short_colnames():

	col_names = []
	for i in range(53):
		col_names.append(idx_to_sen(i))

	return col_names

def get_xmv():

	xmv_attack_numbers = ['a1', 'a2', 'a3', 'a4', 'a6', 'a7', 'a8', 'a10', 'a11']
	return xmv_attack_numbers

def get_non_pid():

	non_pid_sensors = ['s5', 's6', 's13', 's16', 's18', 's19', 's20']
	return non_pid_sensors

def get_pid():
	
	pid_sensors = ['s1', 's2', 's3', 's4', 's7', 's8', 's9', 's10', 's11', 's12', 's14', 's15', 's17', 's23', 's25', 's40']
	return pid_sensors

def get_skip_list():
	
	skip_list = [
		'cons_p2s_s4', 
		'cons_p2s_s9',
		'cons_p2s_a11', 
		'cons_p3s_s4', 
		'cons_p3s_s9',
		'cons_p3s_a11', 
		'cons_p5s_s4', 
		'cons_p5s_s9',
		'cons_p5s_a11', 
		'cons_p5s_s3', 
		'cons_p5s_s17',
		'line_p3s_s9',
		'line_p5s_s9',
		'line_p5s_a11',
		]
	
	return skip_list

def get_footer_list(patterns=None, mags=None, locations=None):

	if locations == None:
		locations = get_pid() + get_xmv()
	elif locations == 'pid':
		locations = get_pid()
	elif locations == 'nonpid':
		locations = get_non_pid()
	elif locations == 'xmv':
		locations = get_xmv()
	
	if patterns is None:
		attack_patterns = ['cons', 'csum', 'line', 'lsum']
	else:
		attack_patterns = patterns

	if mags is None:
		attack_mags = ['p2s', 'm2s', 'p3s', 'p5s']
	else:
		attack_mags = mags

	footers = []
	for am in attack_mags:
		for ap in attack_patterns:
			for loc in locations:
					footer = f'{ap}_{am}_{loc}'
					if footer not in get_skip_list():
						try:
							pd.read_csv(f"tep-attacks/matlab/TEP_test_{footer}.csv", dayfirst=True)
							footers.append(footer)
						except FileNotFoundError:
							continue

	return footers

def find_all(str, ch):
	cuts = []
	for i, ltr in enumerate(str):
		if ltr == ch:
			cuts.append(i)
	return cuts

def lime_exp_to_feature(instexp, count, unique=False):

	choice_string = instexp.as_list()[count][0]
	value = instexp.as_list()[count][1]

	# When parsing, there are 3 possible patterns:
	# XXX < 1.0 
	# XXX > 1.0
	# 0.5 > XXX > 1.0

	# First case
	lt_cuts = find_all(choice_string, '<')
	gt_cuts = find_all(choice_string, '>')

	if len(lt_cuts) == 1:
		cut = lt_cuts[0]
		choice = choice_string[0:cut-1]
	elif len(gt_cuts) == 1:
		cut = gt_cuts[0]
		choice = choice_string[0:cut-1]
	elif len(lt_cuts) == 2:
		choice = choice_string[lt_cuts[0]+2:lt_cuts[1]-1]
	elif len(gt_cuts) == 2:
		choice = 'XXX'
	else:
		print(f'Could not parse the following lime output: {choice_string}')

	# Parse the history from the sensor
	time_cut = find_all(choice, '_')[0]
	sensor = int(choice[0:time_cut])
	history = int(choice[time_cut+3:])

	return sensor, history, value

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

def sen_to_idx(sensor):

	sensor_type = sensor[0]
	sensor_value = int(sensor[1:])

	if sensor_type == 'a':
		return sensor_value + 40
	elif sensor_type == 's':
		return sensor_value - 1

def idx_to_sen(idx):

	if idx > 40:
		return f'a{idx-40}'

	return f's{idx+1}'

# Convert a set of (scores, true label) to its ranking
def scores_to_rank(scores, true_idx):

	rulerank = len(scores) - np.where(np.argsort(scores) == true_idx)[0][0]
	return rulerank

# Get a set of relevance_scores and predicted scores
def calc_dcg(pred_scores, rel_scores, p=10):

	# Take the indices of the highest p predicted scores
	rankings = np.argsort(pred_scores)[::-1]
	total_score = 0

	for i in range(p):
		rank_i_idx = rankings[i]
		score = (np.power(2, rel_scores[rank_i_idx]) - 1) / np.log2(i + 2)
		total_score += score

	return total_score

# Same as DCG, but include normalization to [0-1] based on ideal score
def calc_ndcg(pred_scores, rel_scores, p=10):
	score = calc_dcg(pred_scores, rel_scores, p)
	ideal_score = calc_dcg(rel_scores, rel_scores, p)
	return score / ideal_score

def attack_footer_to_sensor_idx(attack_footer):

	splits = attack_footer.split("_")
	sensor_type = splits[2][0]
	sensor_value = int(splits[2][1:])

	if sensor_type == 'a':
		return sensor_value + 40
	elif sensor_type == 's':
		return sensor_value - 1
	else:
		print(f'Something wrong! Found sensor_type {sensor_type}')
		exit()

	return -1

def load_tep_attack(dataset_name, attack_footer, scaler=None, no_transform=False, verbose=1):

	if verbose > 0:
		print('Loading {} test data...'.format(dataset_name))

	if scaler is None:
		if verbose > 0:
			print('No scaler provided, loading from models directory.')
		scaler = pickle.load(open(f'models/{dataset_name}_scaler.pkl', "rb"))

	if dataset_name == 'TEP':
		df_test = pd.read_csv(f"tep-attacks/matlab/TEP_test_{attack_footer}.csv", dayfirst=True)
		sensor_cols = [col for col in df_test.columns if col not in ['Atk']]
		target_col = 'Atk'

	else:
		print('This script is meant for TEP only.')
		return

	# scale sensor data
	if no_transform:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
	else:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
	
	Ytest = df_test[target_col]

	return Xtest.values, Ytest.values, Xtest.columns

def quant_to_sample_ui(scores, sensor_vals, sensor_cols, attack_code):

	rankings = np.argsort(scores)[::-1]
	alarm_levels = []
	
	for i in range(len(scores)):

		if i in rankings[0:3]:
			alarm_levels.append('<mark style="background-color: red;">HIGH</mark>')
			#alarm_levels.append('HIGH')
		elif i in rankings[3:10]:
			alarm_levels.append('<mark style="background-color: yellow;">MEDIUM</mark>')
			#alarm_levels.append('MEDIUM')
		else:
			alarm_levels.append('LOW')

	data = {'sensor/actuator': sensor_cols, 'current_value': sensor_vals, 'anomaly_score': scores, 'alarm_level': alarm_levels}
	df = pd.DataFrame(data=data)

	html_file = open(f'report_{attack_code}.html', 'w')
	html_file.write(df.sort_values('anomaly_score', ascending=False).to_html(escape=False))
	html_file.close()

	return
