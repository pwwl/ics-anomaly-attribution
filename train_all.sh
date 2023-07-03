#!/usr/bin/env bash
set -euxo pipefail
datasets=('SWAT' 'WADI' 'TEP')
models=('CNN' 'GRU' 'LSTM')
for md in ${models[@]}; do
    for ad in ${datasets[@]}; do
        python main_train.py "${md}" "${ad}" --train_params_epochs $1
    done
done