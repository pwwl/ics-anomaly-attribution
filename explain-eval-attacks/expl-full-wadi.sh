#!/usr/bin/env bash
layers=(2)
units=(64)
for ll in ${layers[@]}; do
	for uu in ${units[@]}; do
		for an in {0..14}; do
			python main_grad_explain_attacks.py CNN WADI "${an}" --cnn_model_params_layers $ll --cnn_model_params_units $uu --cnn_model_params_history 50 --explain_params_methods SM --run_name results_ns1
			python main_grad_explain_attacks.py LSTM WADI "${an}" --lstm_model_params_layers $ll --lstm_model_params_units $uu --lstm_model_params_history 50 --explain_params_methods SM --run_name results_ns1
			python main_grad_explain_attacks.py GRU WADI "${an}" --gru_model_params_layers $ll --gru_model_params_units $uu --gru_model_params_history 50 --explain_params_methods SM --run_name results_ns1
		done
	done
done


