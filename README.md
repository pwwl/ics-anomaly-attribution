<!-- 
   Copyright 2023 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

# Attributions for ML-based ICS Anomaly Detection

This repository contains code for the paper: "Attributions for ML-based ICS anomaly detection: From theory to practice", to appear at the 31st Network and Distributed System Security Symposium (NDSS 2024).

### Bibtex

[![DOI](https://zenodo.org/badge/658993985.svg)](https://zenodo.org/badge/latestdoi/658993985)

```
@inproceedings{icsanomaly:ndss2024,
  title = {Attributions for {ML}-based {ICS} Anomaly Detection: {From} Theory to Practice},
  author = {Clement Fung and Eric Zeng and Lujo Bauer},
  booktitle = {Proceedings of the 31st Network and Distributed System Security Symposium},
  publisher = {Internet Society},
  year = 2024,
}  
```

## Table of Contents

### Core Experiment Workflow
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Workflow 1 - CNN on SWaT Dataset](#workflow-1---cnn-on-swat-dataset)
- [Workflow 2 - CNN on TEP Dataset](#workflow-2---cnn-on-tep-dataset)

### Reference information
- [Overview of the Repository](#overview-of-the-repository)
- [Dataset Descriptions](#datasets)
- [Model Descriptions](#models)
- [Parameters](#parameters)

## Core Experiment Workflow

### Requirements

This project uses Python3 and Tensorflow 1, which requires 64-bit Python 3.7 (or lower).
The best way to get set up is with a Python virtual environment (we recommend using conda). If you don't already have Anaconda3 installed, find your machine's installer [here](https://www.anaconda.com/download#downloads) and complete the installation process.
We also recommend using a Linux machine to avoid problems with executing Bash scripts.

Our primarily development environment was a commodity desktop with 32 GB RAM, using Ubuntu 20.04.
To store all packages, datasets, trained models, and output files, approximately 10 GB of storage is sufficient.
If downloading and testing on the [full set of TEP manipulations](https://doi.org/10.1184/R1/23805552), another 50 GB is required.

### Installation

If you are on a Windows machine, we recommend using the Anaconda Prompt that came with the Anaconda3 installation. Otherwise, simply use the terminal.
Ensure that conda is up to date with:
```sh
conda update conda --all
```
Create a Python 3.7 virtual environment called venv and activate it:
```sh
conda create --name venv python==3.7
conda activate venv
```
Clone this repository with:
```sh
git clone https://github.com/pwwl/ics-anomaly-attribution.git
```
Navigate into the repository and install the requirements for this project:
```sh
cd ics-anomaly-attribution
pip install -r requirements.txt
```

### Data Setup

This repository is configured for three datasets: `TEP`, `SWAT`, and `WADI`.

For convenience, the TEP training dataset is included. 
The raw SWaT and WADI datasets need to be requested through the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

For instructions on how to setup and process the raw datasets, see the associated README files in the `data` directory.

### Workflow 1 - CNN on SWaT Dataset

This workflow will walk you through training a CNN model on the SWaT dataset, as well as generating explanations on a singular attack. Ensure you have retrieved the dataset as mentioned [here](#using-the-datasets).

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a CNN model on the SWaT dataset:
```sh
python main_train.py CNN SWAT --train_params_epochs 10
```
This will utilize a default configuration of two layers, a history length of 50, a kernel size of 3, and 64 units per layer for the CNN model. See detailed explanations for main_train.py parameters [here](#parameters).

Next, use the CNN model to make predictions on the SWaT test dataset, and save the corresponding MSES.
```sh
python save_model_mses.py CNN SWAT
```

Additionally, use the CNN model to make predictions on the SWaT test dataset, and save the corresponding detection points. 
The default detection threshold is set at the 99.95-th percentile validation error.
```sh
python save_detection_points.py --md CNN-SWAT-l2-hist50-kern3-units64-results
```

Run attribution methods for SWaT attack #1, using the scripts in the `explain-eval-attacks` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-attacks
python main_grad_explain_attacks.py CNN SWAT 1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_attacks.py CNN SWAT 1 --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_attacks.py CNN SWAT 1 --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-swat.sh` are provided for reference.
Note: running the explanations may take anywhere from 20 minutes to two hours depending on your machine, so stay patient!
Additionally, depending on your shell configuration, you may need to change `python` to `python3` in the Bash scripts. If you are on Windows, you may also need to install and run [dos2unix](https://sourceforge.net/projects/dos2unix/) on the Bash scripts if you encounter errors with `\r` characters.

Finally, rank the attribution methods for SWaT attack #1: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties.py 1 --md CNN-SWAT-l2-hist50-kern3-units64-results
```

**Note: All core experiments in this work follow the same workflow. To fully reproduce our results and generate plots, experiments must be run on all models (CNN, GRU, LSTM), all attacks/manipulations in all datasets (SWAT, WADI, TEP), and against all attribution methods (CF, SM, SG, IG, EG, LIME, SHAP, LEMNA).**

> For examples of how to train a GRU or LSTM model on SWaT dataset, please see the provided guides for [GRU](README-alt-workflow.md#workflow-1b---gru-on-swat-dataset) and [LSTM](README-alt-workflow.md#workflow-1c---lstm-on-swat-dataset) respectively. The command line arguments differ slightly.

### Workflow 2 - CNN on TEP Dataset

We provide another example that evaluates attribution methods on our synthetic manipulations: this workflow is similar to workflow 1 but is performed on the TEP dataset. 
Because of differences in how features are internally represented between datasets, the workflow uses slightly modified scripts specifically for dealing with the TEP dataset. 
This will also generate explanations on a singular TEP attack. Ensure you have retrieved the training dataset as mentioned [here](#using-the-datasets).
The sample attack used for this workflow is provided in `tep-attacks/matlab/TEP_test_cons_p2s_s1.csv`, which is a constant, two-standard-deviation manipulation on the first TEP sensor.

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a CNN model on the TEP dataset:
```sh
python main_train.py CNN TEP --train_params_epochs 10
```

Next, use the CNN model to make predictions on the TEP manipulation and save the corresponding MSES.
```sh
python save_model_mses.py CNN TEP
```

Additionally, use the CNN model to make predictions on the TEP manipulation and save the corresponding detection points. 
```sh
python save_detection_points.py --md CNN-TEP-l2-hist50-kern3-units64-results
```

Run attribution methods for the TEP manipulation, using the scripts in the `explain-eval-manipulations` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-manipulations
python main_tep_grad_explain.py CNN TEP cons_p2s_s1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_manipulations.py CNN TEP --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_manipulations.py CNN TEP --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-tep.sh` are provided for reference.

Note: running the explanations may take anywhere from 20 minutes to two hours depending on your machine, so stay patient!
Additionally, depending on your shell configuration, you may need to change `python` to `python3` in the Bash scripts. If you are on Windows, you may also need to install and run [dos2unix](https://sourceforge.net/projects/dos2unix/) on the Bash scripts if you encounter errors with `\r` characters.

Finally, rank the attribution methods for the TEP manipulation: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties_tep.py --md CNN-TEP-l2-hist50-kern3-units64-results
```

> For examples of how to train a GRU or LSTM model on TEP dataset, please see the provided guides for [GRU](README-alt-workflow.md#workflow-2b---gru-on-tep-dataset) and [LSTM](README-alt-workflow.md#workflow-2c---lstm-on-tep-dataset) respectively. The command line arguments differ slightly.

## Reference Information

### Overview of the Repository

- [detector](https://github.com/pwwl/ics-anomaly-attribution/tree/main/detector): 
    - `detector.py`: core definition for detector objects
    - `cnn.py`: model definition for convolutional neural network (CNN)
    - `gru.py`: model definition for gated recurrent unit (GRU)
    - `lstm.py`: model definition for long-short-term memory (LSTM)
- [explain-eval-attacks](https://github.com/pwwl/ics-anomaly-attribution/tree/main/explain-eval-attacks): 
    - `main_bbox_explain_attacks.py`: runner script to compute blackbox attributions on SWAT/WADI datasets
    - `main_grad_explain_attacks.py`: runner script to compute gradient-based attributions on SWAT/WADI datasets
    - `expl-full-bbox.sh`: convenience script to run `main_bbox_explain_attacks.py` for SHAP and LENMA
    - `expl-full-swat.sh`: convenience script to run `main_grad_explain_attacks.py` for SWAT dataset
    - `expl-full-wadi.sh`: convenience script to run `main_grad_explain_attacks.py` for WADI dataset
- [explain-eval-manipulations](https://github.com/pwwl/ics-anomaly-attribution/tree/main/explain-eval-manipulations):
    - `main_bbox_explain_manipulations.py`: runner script to compute blackbox attributions on TEP dataset
    - `main_tep_grad_explain.py`: runner script to compute gradient-based attributions on TEP dataset
    - `expl-full-bbox.sh`: convenience script to run `main_bbox_explain_manipulations.py` for SHAP and LENMA
    - `expl-full-tep.sh`: convenience script to run `main_tep_explain_explain.py`
- [live_bbox_explainer](https://github.com/pwwl/ics-anomaly-attribution/tree/main/live_bbox_explainer):
    - `score_generator.py`: API helper to run blackbox attributions
- [live_grad_explainer](https://github.com/pwwl/ics-anomaly-attribution/tree/main/live_grad_explainer):
    - `explainer.py`: core definition for gradient-based explainer object
    - `expected_gradients_mse_explainer.py`: definition for expected gradients explainer object
    - `integrated_gradients_explainer.py`: definition for integrated gradients explainer object
    - `integrated_gradients_mse_explainer.py`: definition for total-MSE integrated gradients explainer object
    - `smooth_grad_explainer.py`: definition for SmoothGrad and saliency map explainer object
    - `smooth_grad_mse_explainer.py`: definition for total-MSE SmoothGrad and saliency map explainer object
- [models](https://github.com/pwwl/ics-anomaly-attribution/tree/main/models): where trained model metadata is stored
    - `results`: default directory for model metadata storage
- [plotting](https://github.com/pwwl/ics-anomaly-attribution/tree/main/plotting):
    - `make_benchmark_plot.py`: script used to create Figure 2 in paper (for reference)
    - `make_stats_plot.py`: script used to generate stats for Table 4 in paper (for reference)
    - `make_timing_plot.py`: script used to create Figure 4 in paper (for reference)
- [pygflasso](https://github.com/pwwl/ics-anomaly-attribution/tree/main/pygflasso):
    - `gflasso.py`: Fused lasso model, used for LEMNA explanation
- [tep-attacks/matlab](https://github.com/pwwl/ics-anomaly-attribution/tree/main/tep-attacks/matlab): contains CSV files corresponding to TEP attacks
    - `TEP_test_cons_ps2_s1.csv`: contains a constant, two-standard-deviation manipulation on TEP sensor #1.
- [utils](https://github.com/pwwl/ics-anomaly-attribution/tree/main/utils):
    - `attack_utils.py`: utility functions for attack parsing
    - `metrics.py`: utility functions for model metrics
    - `tep_plot_utils.py`: utility functions for plotting, specific to TEP
    - `tep_utils.py`: utility functions for TEP dataset
    - `utils.py`: micellaneous utility functions
- Primary Workflow Scripts
    - `main_train.py`: trains ICS anomaly detection models
    - `save_model_mses.py`: saves MSEs over testing datasets
    - `save_detection_points.py`: saves detection points over testing datasets, used for timing strategies
    - `main_feature_properties.py`: evaluates attribution methods for SWAT/WADI by ranking their scores, at various timing strategies
    - `main_feature_properties_tep.py`: evaluates attribution methods for TEP by ranking their scores, at various timing strategies
- Additional Scripts
    - `make-dirs.sh`: creates needed directories to be populated by files generated by Main Scripts
    - `setup_run_name.sh`: creates a directory in `models/` corresponding to a run name
    - `train-all.sh`: trains all model types on all datasets using `main_train.py`
    - `data_loader.py`: Loads required train and test datasets
    - `main_benchmark.py`: performs the synthetic benchmark tests, described in Section VII-A in paper (not a core contribution)


### Datasets

Three datasets are supported:

* [Secure Water Treatment Plant](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/) (`SWAT`)
    * A 51-feature, 6-stage water treatment process, collected from a water plant testbed in Singapore.
    * Provided by the SUTD iTrust website.
* [Water Distribution](https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/) (`WADI`)
    * A 123 feature dataset of a water distribution system. collected from a water plant testbed in Singapore.
    * Like SWAT, needs to be downloaded from the SUTD iTrust website.
* [Tennessee Eastman Process](https://depts.washington.edu/control/LARRY/TE/download.html) (`TEP`)
    * A 53 feature dataset of a chemical process, collected from a public MATLAB simulation environment.
    * Testing data for this dataset was created by modifying the simulator and systematically injecting manipulations into the process. 
    * The modified simulator is [publicly available](https://github.com/pwwl/tep-attack-simulator).

### Models

We currently support three types of models, all using the [Keras Model API](https://keras.io/models/model/).

* 1-D Convolutional Neural Networks (`CNN`)
    * Deep learning models that use 1 dimensional convolutions (across the time dimension) to summarize temporal patterns in the data. These temporal patterns are stored as a trainable kernel matrix, which is used during the convolution step to identify such patterns. [Read more](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)
* Long Short Term Memory (`LSTM`)
    * Deep learning models that are similar to CNNs: they provide analysis of temporal patterns over the time dimension.  However, the primary difference is that LSTMs do not fix the size of the kernel convoluation window, and thus allow for arbitrarily long patterns to be learned. [Read more](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* Gated Recurrent Units (`GRU`)
    * Deep learning models that provide similar functionality to LSTMs through gates, but use much less state/memory. As a result, they are quicker to train and use, and provide similarly strong performance.

### Parameters

The argparse library is used in most scripts, which can be run with the `--help` flag to display all mandatory and required arguments to the script. Here are detailed accounts of the parameters for each script ran in the workflows.

- `main_train.py`: [Model Parameters](#model-parameters), [Training Parameters](#training-parameters), [Other Parameters](#other-parameters)
- `save_model_mses.py`: [Model Parameters](#model-parameters), [Other Parameters](#other-parameters), [Metrics Parameter](#metrics-parameter)
- `save_detection_points.py`: [Specific Model Parameter](#specific-model-parameter) 
- `explain-eval-attacks/main_bbox_explain_attacks.py`: [Model Parameters](#model-parameters), [Other Parameters](#other-parameters), [Specific Attack Parameter](#specific-attack-parameter), [BBox Parameters](#bbox-parameters)
- `explain-eval-attacks/main_grad_explain_attacks.py`: [Model Parameters](#model-parameters), [Other Parameters](#other-parameters), [Specific Attack Parameter](#specific-attack-parameter), [Grad Parameters](#grad-parameters)
- `explain-eval-manipulations/main_bbox_explain_manipulations.py`: [Model Parameters](#model-parameters), [Other Parameters](#other-parameters), [BBox Parameters](#bbox-parameters)
- `explain-eval-manipulations/main_tep_grad_explain.py`: [Model Parameters](#model-parameters), [Other Parameters](#other-parameters), [Specific Attack Parameter](#specific-attack-parameter), [Grad Parameters](#grad-parameters)
- `main_feature_properties.py`: [Specific Model Parameter](#specific-model-parameter), [Specific Attack Parameter](#specific-attack-parameter)
- `main_feature_properties_tep.py`: [Specific Model Parameter](#specific-model-parameter)

#### Model Parameters

| Name      | Description | Default |
| ----- | ---- | --- |
| --cnn_model_params_units | The number of units in each layer of the CNN. | 64 |
| --cnn_model_params_history | The total size of the prediction window used. When predicting on an instance, this tells the model how far back in time to use in prediction. | 50 |
| --cnn_model_params_layers | The number of CNN layers to use. | 2 |
| --cnn_model_params_kernel | The size of the 1D convolution window used when convolving over the time window. | 3 |
| --lstm_model_params_units   | The number of units in each layer of the LSTM. | 64 |
| --lstm_model_params_history | The total size of the prediction window used. When predicting on an instance, this tells the model how far back in time to use in prediction. | 50 |
| --lstm_model_params_layers   | The number of LSTM layers to use. | 2 |
| --gru_model_params_unit   | The number of units in each layer of the GRU. | 64 |
| --gru_model_params_history | The total size of the prediction window used for the GRU. When predicting on an instance, this tells the model how far back in time to use in prediction. | 50 |
| --gru_model_params_layers'  | The number of GRU layers to use. | 2 |

#### Training Parameters

| Name      | Description | Default |
| ----- | ---- | --- |
| --train_params_epochs | The number of times to go over the training data | 100 |
| --train_params_batch_size | Batch size when training. Note: MUST be larger than all history/window values given. | 512 |
| --train_params_no_callbacks | Removes callbacks like early stopping | False |

#### Other Parameters

| Name      | Description | Default |
| --- | --- | --- |
| model | Type of model to use (CNN, GRU, or LSTM) | CNN |
| dataset | Dataset name to use (SWAT, WADI, or TEP) | TEP |
| --gpus | Which GPUS to use during training and evaluation? This should be specified as a GPU index value, as it is passed to the environment variable `CUDA_VISIBLE_DEVICES`. | None |
| --run_name | If provided, stores all models in the associated `run_name` directory. Note: use `setup_run_name.sh` to create the desired `models/run_name` directory. | result |

#### Metrics Parameter

| Name      | Description | Default |
| ----- | ---- | --- |
| --detect_params_metrics | Metrics to look over (at least one required). | F1 |

#### Specific Model Parameter

| Name      | Description | Default |
| --- | ---- | --- |
| --md | Specifies an exact model to use. Format as `model-dataset-layers-history-kernel-units-runname` if model type is CNN, format as `model-dataset-layers-history-units-runname` otherwise (at least one required). | None |

#### Specific Attack Parameter

| Name      | Description | Default |
| --- | --- | --- |
| attack | Specific attack number to use (at least one required) | None |

#### Blackbox Attribution Parameters

| Name      | Description | Default |
| ----- | ---- | --- |
| --explain_params_methods | Select the attribution methods(s) to use: raw MSE (MSE), LIME, SHAP, or LEMNA | MSE |
| --num_samples | Number of samples | 5 |

#### Gradient Attribution Parameters

| Name      | Description | Default |
| ----- | ---- | --- |
| --explain_params_methods | Select the attribution method(s) to use: saliency map (SM), SmoothGrad (SG), integrated gradients (IG), expected gradients (EG) | SM |
| --explain_params_use_top_feat | Explain based off top MSE feature, rather than entire MSE | False |
| --num_samples | Number of samples | 5 |
