#!/usr/bin/env bash
datasets=('TEP')
for ad in ${datasets[@]}; do
	python3 main_bbox_explain_manipulations.py CNN "${ad}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods SHAP --run_name results --num_samples 150
	python3 main_bbox_explain_manipulations.py CNN "${ad}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods LEMNA --run_name results --num_samples 150
	# python3 main_bbox_explain_manipulations.py GRU "${ad}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --explain_params_methods SHAP --run_name results --num_samples 150
	# python3 main_bbox_explain_manipulations.py GRU "${ad}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --explain_params_methods LEMNA --run_name results --num_samples 150
	# python3 main_bbox_explain_manipulations.py LSTM "${ad}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --explain_params_methods SHAP --run_name results --num_samples 150
	# python3 main_bbox_explain_manipulations.py LSTM "${ad}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --explain_params_methods LEMNA --run_name results --num_samples 150
done