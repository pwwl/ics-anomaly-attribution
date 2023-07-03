#!/usr/bin/env bash

# Full set:
#types=('p2s' 'm2s' 'p3s' 'p5s')
#locations=('s1' 's2' 's3' 's7' 's8' 's10' 's11' 's12' 's14' 's15' 's17' 's23' 's25' 's40' 'a1' 'a2' 'a3' 'a4' 'a6' 'a7' 'a8' 'a10' 's4' 's9' 'a11')

# Minimal example:
types=('p2s')
locations=('s1')
patterns=('cons')

for ap in ${patterns[@]}; do
	for an in ${locations[@]}; do
		for at in ${types[@]}; do
			python3 main_tep_grad_explain.py CNN TEP "${ap}_${at}_${an}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods SM --run_name results --num_samples 150
			# python3 main_tep_grad_explain.py GRU TEP "${ap}_${at}_${an}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --explain_params_methods SM --run_name results --num_samples 150
			# python3 main_tep_grad_explain.py LSTM TEP "${ap}_${at}_${an}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --explain_params_methods SM --run_name results --num_samples 150
		done
	done
done