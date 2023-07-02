#!/usr/bin/env bash
datasets=('SWAT') #'WADI')
attacks=(1)
for ad in ${datasets[@]}; do
	for an in ${attacks[@]}; do
		/mnt/c/Users/steph/anaconda3/envs/venv/python.exe main_bbox_explain_attacks.py CNN "${ad}" "${an}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods SHAP --run_name results
		/mnt/c/Users/steph/anaconda3/envs/venv/python.exe main_bbox_explain_attacks.py CNN "${ad}" "${an}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods LEMNA --run_name results
		/mnt/c/Users/steph/anaconda3/envs/venv/python.exe main_bbox_explain_attacks.py CNN "${ad}" "${an}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods SHAP --run_name results --num_samples 150
		/mnt/c/Users/steph/anaconda3/envs/venv/python.exe main_bbox_explain_attacks.py CNN "${ad}" "${an}" --cnn_model_params_layers 2 --cnn_model_params_units 64 --cnn_model_params_history 50 --explain_params_methods LEMNA --run_name results --num_samples 150
		#python main_bbox_explain_attacks.py GRU "${ad}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --explain_params_methods SHAP --run_name results_ns1
		#python main_bbox_explain_attacks.py GRU "${ad}" --gru_model_params_layers 2 --gru_model_params_units 64 --gru_model_params_history 50 --explain_params_methods LEMNA --run_name results_ns1
		#python main_bbox_explain_attacks.py LSTM "${ad}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --explain_params_methods SHAP --run_name results_ns1
		#python main_bbox_explain_attacks.py LSTM "${ad}" --lstm_model_params_layers 2 --lstm_model_params_units 64 --lstm_model_params_history 50 --explain_params_methods LEMNA --run_name results_ns1
	done
done