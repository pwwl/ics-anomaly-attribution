#!/usr/bin/env bash

set -euxo pipefail

layers=(2)
units=(64)

for ll in ${layers[@]}; do
	for uu in ${units[@]}; do
		for an in {0..31}; do
			python3 main_grad_explain_attacks.py CNN SWAT "${an}" --cnn_model_params_layers $ll --cnn_model_params_units $uu --cnn_model_params_history 50 --explain_params_methods SM --run_name results_ns1
			python3 main_grad_explain_attacks.py LSTM SWAT "${an}" --lstm_model_params_layers $ll --lstm_model_params_units $uu --lstm_model_params_history 50 --explain_params_methods SM --run_name results_ns1
			python3 main_grad_explain_attacks.py GRU SWAT "${an}" --gru_model_params_layers $ll --gru_model_params_units $uu --gru_model_params_history 50 --explain_params_methods SM --run_name results_ns1
		done
	done
done

