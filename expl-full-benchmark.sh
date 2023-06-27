#!/usr/bin/env bash

set -euxo pipefail

datasets=('SWAT')
methods=('CF-Add' 'CF-Sub' 'SM' 'SG' 'IG' 'EG' 'LIME' 'SHAP' 'LEMNA')

for dd in ${datasets[@]}; do
	for meth in ${methods[@]}; do
		python3 main_benchmark.py CNN "${dd}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_kernel 3 --cnn_model_params_history 50 --run_name results_ns1 --explain_params_method "${meth}"
	done
done

for dd in ${datasets[@]}; do
	for meth in ${methods[@]}; do
		python3 main_benchmark.py GRU "${dd}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --run_name results_ns1 --explain_params_method "${meth}"
		python3 main_benchmark.py LSTM "${dd}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --run_name results_ns1 --explain_params_method "${meth}"
	done
done

