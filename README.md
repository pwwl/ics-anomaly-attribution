<!-- 
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
-->

# Framework for Training and Evaluating ML-based ICS Anomaly Detectors 
    
## Table of Contents

First time here?

- [Installation](#installation)
- [Run Quickstart First Time Demo](#first-time-demo)

Reference Information

- [Dataset Descriptions](#datasets)
- [Model Descriptions](#models)

Usage

- [Execution](#execution)
- [Training](#training)
- [Explanations](#explanations)

## Installation

This project uses virtualenv to manage dependencies. It's a lightweight, simple solution that takes only a few instructions to setup. Some people prefer conda which is a bit more heavy-weight, but will also work for this setup.

Setup:
1. Install virtualenv: `sudo apt install virtualenv` (ubuntu) or perhaps `brew install virtualenv` (if on Mac)
2. Download the latest release version of the code: https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/releases
3. In this project's root directory (`cd ics-ml-evasion-tool`), start a new virtual environment. `virtualenv -p python3 venv`
    4. Python3 is a dependency and must be used in this project
5. Activate the environment: `source venv/bin/activate`
6. Install the frozen requirements packages: `pip3 install -r requirements.txt`

## Datasets

Three datasets are supported:

* [Secure Water Treatment Plant](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/) (`SWAT`)
    * A 6 stage water treatment process, collected from a water plant testbed in Singapore.
    * Contains 77 sensors/actuators, and 6 labelled cyber-attacks
    * Download instructions are in the `data/SWAT` folder.
* [Water Distribution](https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/) (`WADI`)
    * A 123 feature (sensors, actuators) dataset of a water distribution system.
    * Like SWAT, needs to be downloaded from the SUTD iTrust website.
    * Some features were missing small number of values, and the script `process_WADI.py` does a manual interpolation of these gaps.
* TEP (`TEP`)
    * TBC

## Models

We currently support three types of models, all using the [Keras Model API](https://keras.io/models/model/).

* 1-D Convolutional Neural Networks (`CNN`)
    * Deep learning models that use 1 dimensional convolutions (across the time dimension) to summarize temporal patterns in the data. These temporal patterns are stored as a trainable kernel matrix, which is used during the convolution step to identify such patterns. [Read more](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)
* Long Short Term Memory (`LSTM`)
    * Deep learning models that are similar to CNNs: they provide analysis of temporal patterns over the time dimension.  However, the primary difference is that LSTMs do not fix the size of the kernel convoluation window, and thus allow for arbitrarily long patterns to be learned. [Read more](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* Gated Recurrent Units (`GRU`)
    * Deep learning models that provide similar functionality to LSTMs through gates, but use much less state/memory. As a result, they are quicker to train and use, and provide similarly strong performance.

## Execution

TBC

## Training

### Model and Training Config (new)

We use the argparse library. 
Running the script with `python3 main_train.py --help` will display all the mandatory and required arguments to the script.
Each configuration value, defaults and documentation are stored in the `utils.py` file.

#### Model parameters

| Name      | Description | Default |
| --- | --- | --- |
| '--cnn_model_params_units' | The number of units in each layer of the CNN. | 32 |
| '--cnn_model_params_history' | The total size of the prediction window used. When predicting on an instance, this tells the model how far back in time to use in prediction. | 200 |
| '--cnn_model_params_layers' | The number of CNN layers to use. | 8 |
| '--cnn_model_params_kernel' | The size of the 1D convolution window used when convolving over the time window. | 2 |
| '--lstm_model_params_units'   | The number of units in each layer of the LSTM. | 512 |
| '--lstm_model_params_history' | The total size of the prediction window used. When predicting on an instance, this tells the model how far back in time to use in prediction. | 100 |
| '--lstm_model_params_layers'   | The number of LSTM layers to use. | 4 |
| '--gru_model_params_units'   | The number of units in each layer of the GRU. | 256 |
| '--gru_model_params_history' | The total size of the prediction window used for the GRU. When predicting on an instance, this tells the model how far back in time to use in prediction. | 100 |
| '--gru_model_params_layers'   | The number of GRU layers to use. | 2 |

#### Training parameters

| Name      | Description | Default |
| --- | --- | --- |
| '--train_params_epochs' | The number of times to go over the training data | 100 |
| '--train_params_batch_size' | Batch size when training. Note: MUST be larger than all history/window values given. | 512 |
| '--train_params_no_callbacks' | If specified, do not use training callbacks like early stopping and learning rate modification. | False |

#### Other 

| Name      | Description | Default |
| --- | --- | --- |
| '--gpus' | Which GPUS to use during training and evaluation? This should be specified as a GPU index value, as it is passed to the environment variable `CUDA_VISIBLE_DEVICES`. | None |
| '--run_name' | If provided, stores all models, plots and npy files in the associated `run_name` directory. Note: If saving plots, npys, or models, each of `models/run_name`, `npys/run_name`, and `plots/run_name` directories must exist. | None |

## Attributions

The main attribution code is stored in `grad_explainer` and `live_explainer`.

#### Explanation config (Needs to be updated)

| Name      | Description | Default |
| --- | --- | --- |
| '--explain_params_methods' | Provides a list of explanation methods to use. Choices are saliency maps (SM), SmoothGrad (SG), integrated gradients (IG), and expected gradients (EG) | SM, SG |
| '--explain_params_use_top_feat' | If specified, use the top feature MSE as the gradient-based function, as opposed to the entire MSE.  | False |
| '--explain_params_threshold' | Percentile detection threshold for selecting candidates in true positives for explanation. Specifying this value implies a window value of 1. Setting a threshold of 0 (default) chooses the stored optimal detection hyperparameters. | 0 (optimal) |

