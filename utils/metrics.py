"""

	 Copyright 2020 Lujo Bauer, Clement Fung

	 Licensed under the Apache License, Version 2.0 (the "License");
	 you may not use this file except in compliance with the License.
	 You may obtain a copy of the License at

			 http://www.apache.org/licenses/LICENSE-2.0

	 Unless required by applicable law or agreed to in writing, software
	 distributed under the License is distributed on an "AS IS" BASIS,
	 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	 See the License for the specific language governing permissions and
	 limitations under the License.

"""

import matplotlib.pyplot as plt
import numpy as np
import pdb

def get(name):

	if name == 'accuracy':
		return accuracy
	elif name == 'precision':
		return precision
	elif name == 'recall':
		return recall
	elif name == 'F1':
		return f1_score
	elif name == 'FB13':
		return fb13_score
	elif name == 'FB31':
		return fb31_score
	elif name == 'TAP':
		return time_aware_precision
	elif name == 'TAR':
		return time_aware_recall
	elif name == 'NAB':
		return numenta
	elif name == 'NE':
		return numenta_early
	elif name == 'EP':
		return early_positives # for measuring the success of numenta (20%)
	elif name == 'FP':
		return false_positive_segments
	elif name == 'TP':
		return true_positive_segments
	elif name == 'SF1':
		return segment_f1
	elif name == 'SPPV':
		return segment_ppv
	elif name == 'STPR':
		return segment_tpr
	elif name == 'SFB13':
		return segment_fb13_score
	elif name == 'SFB31':
		return segment_fb31_score
	else:
		print('Metric {} was not found. Default to F1.'.format(name))

	return f1_score

def accuracy(ypred, ytrue):
	
	if np.sum(ypred) == 0:
		return 0

	return np.mean(ypred == ytrue)

def precision(ypred, ytrue):
	
	if np.sum(ypred) == 0:
		return 0

	return np.mean(ypred[ypred.astype(bool)] == ytrue[ypred.astype(bool)])

def recall(ypred, ytrue):
	return np.mean(ypred[ytrue.astype(bool)] == ytrue[ytrue.astype(bool)])

def f1_score(ypred, ytrue):
	return fb_score(ypred, ytrue, 1)

# F1, recall 3 times more important
def fb13_score(ypred, ytrue):
	return fb_score(ypred, ytrue, 3)

# F1, precision 3 times more important
def fb31_score(ypred, ytrue):
	return fb_score(ypred, ytrue, 1/3)

def fb_score(ypred, ytrue, beta=1):
	
	if np.sum(ypred) == 0:
		return 0

	prec = precision(ypred, ytrue) 
	rec = recall(ypred, ytrue) 

	if prec == 0 and rec == 0:
		return 0

	return (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)

# Utility function: converts a 0-1 timeseries into its segment [(start, end)] representation
def to_segments(sequence):

	starts = []
	ends = []
	length = len(sequence)

	for i in np.arange(length):
		 
		# Record any 0-1 transition, or the start of the signal
		if (sequence[i] and i == 0) or (sequence[i] and not sequence[i-1]):
			starts.append(i)

		# Record any 1-0 transition, or the end of the signal
		if (sequence[i] and i == (length - 1)) or (sequence[i] and not sequence[i+1]):
			ends.append(i)

	return starts, ends

def early_positives(ypred, ytrue):
	
	score = 0
	position_bias = 0.2
	starts, ends = to_segments(ytrue)

	for j in range(len(starts)):
		
		# Pred region is RELATIVE to the labelled anomaly
		pred_region = ypred[starts[j]:ends[j]]
		detect_idx = np.where(pred_region)[0]

		if len(detect_idx) > 0:
			
			# Find the value on the sigmoid function (first detection)
			first_detect = np.min(detect_idx)

			# Check if first detection is before the cutoff
			if first_detect <= int(position_bias * len(pred_region)):
				print(f'detection at {first_detect} out of {len(pred_region)} is early')
				score += 1
			else:
				print(f'detection at {first_detect} out of {len(pred_region)}')

	return score

def numenta_early(ypred, ytrue):
	return numenta(ypred, ytrue, kappa=10, position_bias=0.2, fp_weight = -0.5)

def numenta(ypred, ytrue, kappa = 5, position_bias = 0.5, fp_weight = -1, fn_weight = -1, tp_weight = 1):

	if len(ypred) != len(ytrue):
		print('Warning! Segment lengths are not the same!! True:{} Pred:{}'.format(len(ytrue), len(ypred)))

	starts, ends = to_segments(ytrue)
	score = 0  

	# Create a segment trace of only FP segments
	ydiff = ypred - ytrue
	ydiff[ydiff == -1] = 0

	dstarts, dends = to_segments(ydiff)
	
	# Each false positive prediction is fp_weight
	score = len(dstarts) * fp_weight  

	for j in range(len(starts)):
		
		# Pred region is RELATIVE to the labelled anomaly
		pred_region = ypred[starts[j]:ends[j]]
		detect_idx = np.where(pred_region)[0]

		# False negative
		if len(detect_idx) == 0:
			sig_value = fn_weight
		else:
			
			# Find the value on the sigmoid function (first detection)
			first_detect = np.min(detect_idx)

			# Calibrate against the center of the sigmoid.
			# Default of 0.5 means that the sigmoid is centered on the anomaly
			# Small values will push the sigmoid closer to the front
			f_value = first_detect - int(position_bias * len(pred_region))

			# Higher kappa = thinner sigmoid
			sig_value = (tp_weight - fp_weight) / (1 + np.exp(kappa * f_value)) - fp_weight
		
			#print(f'For detection at {first_detect} out of {len(pred_region)}, {sig_value}')

		score += sig_value * tp_weight

	# Take the mean of Numenta scores
	score /= (len(starts) + len(dstarts))

	return score

def time_aware_recall(ypred, ytrue, alpha=0.5):

	if len(ypred) != len(ytrue):
		print('Warning! Segment lengths are not the same!! True:{} Pred:{}'.format(len(ytrue), len(ypred)))

	starts, ends = to_segments(ytrue)
	score = 0

	for j in range(len(starts)):
		pred_region = ypred[starts[j]:ends[j]]

		if np.any(pred_region):
			existence = 1
		else:
			existence = 0

		score += alpha * existence + (1 - alpha) * np.mean(pred_region)
		
	# Take the mean of the TA-recalls
	score /= len(starts)

	return score

def time_aware_precision(ypred, ytrue):

	if len(ypred) != len(ytrue):
		print('Warning! Segment lengths are not the same!! True:{} Pred:{}'.format(len(ytrue), len(ypred)))

	starts, ends = to_segments(ypred)
	score = 0

	if len(starts) == 0:
		return -1 # No predictions made.

	for j in range(len(starts)):
		pred_region = ytrue[starts[j]:ends[j]+1]
		#print(len(pred_region))
		score += np.mean(pred_region)
		
	# Take the mean of the TA-precisions
	score /= len(starts)

	return score

def segment_ppv(ypred, ytrue):

	tps = true_positive_segments(ypred, ytrue)
	fps = false_positive_segments(ypred, ytrue)

	if tps + fps == 0:
		return 0

	return tps / (tps + fps)

def segment_tpr(ypred, ytrue):

	true_starts, _ = to_segments(ytrue)
	segment_recall = true_positive_segments(ypred, ytrue) / len(true_starts)

	return segment_recall

# Segment F1, precision 3 times more important
def segment_f1(ypred, ytrue):
	return segment_fb_score(ypred, ytrue, 1)

# Segment F1, recall 3 times more important
def segment_fb13_score(ypred, ytrue):
	return segment_fb_score(ypred, ytrue, 3)

# Segment F1, precision 3 times more important
def segment_fb31_score(ypred, ytrue):
	return segment_fb_score(ypred, ytrue, 1/3)

# Segment F1, precision 3 times more important
def segment_fb_score(ypred, ytrue, beta=1):

	segment_recall = segment_tpr(ypred, ytrue)
	segment_precision = segment_ppv(ypred, ytrue)

	if segment_recall + segment_recall == 0:
		return 0

	return (1 + beta**2) * (segment_precision * segment_recall) / (beta**2 * segment_precision + segment_recall)

def true_positive_segments(ypred, ytrue):

	if len(ypred) != len(ytrue):
		print('Warning! Segment lengths are not the same!! True:{} Pred:{}'.format(len(ytrue), len(ypred)))

	starts, ends = to_segments(ytrue)
	score = 0

	for j in range(len(starts)):
		pred_region = ypred[starts[j]:ends[j]+1]
		
		if np.any(pred_region):
			score += 1

	return score

def false_positive_segments(ypred, ytrue):

	if len(ypred) != len(ytrue):
		print('Warning! Segment lengths are not the same!! True:{} Pred:{}'.format(len(ytrue), len(ypred)))

	starts, ends = to_segments(ypred)
	score = 0

	if len(starts) == 0:
		return -1 # No predictions made.

	for j in range(len(starts)):
		pred_region = ytrue[starts[j]:ends[j]+1]
		
		if np.any(pred_region == 1):
			continue
		else:
			score += 1

	return score
