#### A collection of various index selector strategies.

import numpy as np
import pdb

HOUR = 2000

def select_by_first_detection(event_detector, Xval, Xtest, quant=0.9995, detection_cutoff=None, window=10):

	n_sensor = Xtest.shape[1]

	if detection_cutoff is None:
		full_val_errors = event_detector.reconstruction_errors(Xval, batches=True)
		detection_cutoff = np.quantile(np.mean(full_val_errors, axis=1), quant)

	full_test_errors = event_detector.reconstruction_errors(Xtest, batches=True)
	full_detection = np.mean(full_test_errors, axis=1) > detection_cutoff

	history = event_detector.params['history']
	att_start = 5*HOUR - history
	att_end = 7*HOUR - history

	## Explore within attack
	attack_detection_idxs = np.where(full_detection[att_start:att_end])[0]
	window_attack_detection_idxs = np.where(np.convolve(full_detection[att_start:att_end], np.ones(window), 'same') // window)[0]

	if len(window_attack_detection_idxs) > 0:
		detect_idx = np.min(window_attack_detection_idxs) + att_start
	else:
		detect_idx = -1

	return detect_idx

