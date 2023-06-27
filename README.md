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
- [Evaluation](#evaluation)
- [Explanations](#explanations)
- [Adversarial Attacks](#adversarial-attacks)

## Installation

This project uses virtualenv to manage dependencies. It's a lightweight, simple solution that takes only a few instructions to setup. Some people prefer conda which is a bit more heavy-weight, but will also work for this setup.

Setup:
1. Install virtualenv: `sudo apt install virtualenv` (ubuntu) or perhaps `brew install virtualenv` (if on Mac)
2. Download the latest release version of the code: https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/releases
3. In this project's root directory (`cd ics-ml-evasion-tool`), start a new virtual environment. `virtualenv -p python3 venv`
    4. Python3 is a dependency and must be used in this project
5. Activate the environment: `source venv/bin/activate`
6. Install the frozen requirements packages: `pip3 install -r requirements.txt`

## First Time Demo

### Prep training data

If you want to run the demos provided, you'll also need to prepare the training data. (In the demo, only BATADAL is used).  

1. From the `data` directory: `cd data/`
2. Unzip the data folder: `tar -xvf BATADAL.tar.gz`

### Run demo on Jupyter

If it's your first time, I recommend jupyter since it's quite easy to use and document in notebook.
If you followed the installation steps above, you should already have jupyter installed and ready to go!  

Start the notebook. From the project root: `jupyter notebook .`   
This will open a browser window with the notebook environment filesystem. Within this environment, double-click to open the `first-time-demo.ipynb` file. The notebook will contain steps to load data, train a ML detector, configure hyper-parameters, and evaluate its performance.

If you need a primer on how notebooks work, you can [review their documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#notebook-user-interface).

## Datasets

Currently, five datasets are supported:

* [Secure Water Treatment Plant](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/) (`SWAT`)
    * A 6 stage water treatment process, collected from a water plant testbed in Singapore.
    * Contains 77 sensors/actuators, and 6 labelled cyber-attacks
    * Download instructions are in the `data/SWAT` folder.
* [Prof. Morris (U. Alabama Huntsville) Gas Pipeline](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) (`gas-pipeline`)
    * Listed as "Dataset 4: New Gas Pipeline"
        * Code is available in ARFF format, I downloaded it, cleaned it, and converted to CSV.
    * Contains 25 features, some sensors/actuators, some network level statistics. 7 types of labelled cyber-attacks
        * The data is a bit messy, many of the attacks are not contiguous and thus, very hard to detect with our ML models 
    * Download instructions are in the `data/gas-pipeline` folder.
* [Battle of the Attack Detection Algorithms](https://www.batadal.net/) (`BATADAL`)
    * A 43 feature (sensors, actuators) dataset of a simulated water distribution network.
    * Dataset is small enough to be zipped and checked into git. No need to download.
* [Water Distribution](https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/) (`WADI`)
    * A 123 feature (sensors, actuators) dataset of a water distribution system.
    * Like SWAT, needs to be downloaded from the SUTD iTrust website.
    * Some features were missing small number of values, and the script `process_WADI.py` does a manual interpolation of these gaps.
* [HIL Augmented ICS](https://github.com/icsdataset/hai) (`HAI`)
    * A new ICS dataset from Feb 2020
    * 59 sensors and actuators, from a simulated thermal and hydropower generation plant
    * Download README is in `data/HAI` folder. Just download and gunzip the CSV files from the github repo.

### Adding a new dataset

The framework currently expects data to be formatted in CSV format, with the first row of the CSV providing the column headers, and one of the columns specifying the attack state (on test data) as a 0/1 integer.

For many of the provided datasets, [a quick script](https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/src/master/data/SWAT/process_SWaT.py) was written to format the dataset, and might be useful for reference.

The easiest way to add a new dataset would be to insert some logic into the `data_loader/load_train_data` and `data_loader/load_test_data` functions, and store the dataset in the `data` folder.

Below is an example for the training data of SWAT:

````
elif dataset_name == 'SWAT':
    df_train = pd.read_csv("data/" + dataset_name + "/SWATv0_train.csv", dayfirst=True)
    sensor_cols = [col for col in df_train.columns if col not in ['Timestamp', 'Normal/Attack']]
````

1. Provide a name for the dataset and insert an `elif` block for it.
    2. The name provided will be the same name used when calling a train function at the top level.
3. Read the dataset using pandas `pd.read_csv()`. This should automatically populate the data frame with the column names in the first row of the CSV.
4. Next, provide a list of columns to filter out of the dataframe, that will not be used for training. For example, many datasets come with labels such as time and date, which we don't want to feed to the ML model. 
    5. Since the training is unsupervised, we also remove the labelled column (if it exists).

For the test data, we do the same steps, but include the target column: 
````
elif dataset_name == 'SWAT':
    df_test = pd.read_csv("data/" + dataset_name + "/SWATv0_test.csv")
    sensor_cols = [col for col in df_test.columns if col not in ['Timestamp', 'Normal/Attack']]
    target_col = 'Normal/Attack'
````

The rest of the `data_loader` will standardize the dataset and do a training/validation split.  
This is a generic step and you should not have to change it when adding a new dataset.

### Dataset Cleaning

Some recent experiments and prior work have suggested using the Kolmogorov-Smirnov test to filter out and remove features whose train-test distributions vary significantly. This technique was proposed for the SWAT dataset in Section V of [this ArXiv paper](https://arxiv.org/abs/1907.01216) and is also locally implemented in `main_data_cleaning.py`.

To use these cleaned versions of the dataset, we have added dataset codes `SWAT-CLEAN` and `WADI-CLEAN`.

## Models

We currently support six types of models, all using the [Keras Model API](https://keras.io/models/model/).

* Autoencoders (`AE`)
    * Compression-based model that reduces the size of the parameters, and expands them back to the original size. In doing so, the compressed embedding stores the elements of the data that are most relevant. This is a commonly used technique for anomaly detection models. [Read more](https://blog.keras.io/building-autoencoders-in-keras.html)
* 1-D Convolutional Neural Networks (`CNN`)
    * Deep learning models that use 1 dimensional convolutions (across the time dimension) to summarize temporal patterns in the data. These temporal patterns are stored as a trainable kernel matrix, which is used during the convolution step to identify such patterns. [Read more](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)
* Long Short Term Memory (`LSTM`)
    * Deep learning models that are similar to CNNs: they provide analysis of temporal patterns over the time dimension.  However, the primary difference is that LSTMs do not fix the size of the kernel convoluation window, and thus allow for arbitrarily long patterns to be learned. [Read more](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* Gated Recurrent Units (`GRU`)
    * Deep learning models that provide similar functionality to LSTMs through gates, but use much less state/memory. As a result, they are quicker to train and use, and provide similarly strong performance.
* Support Vector Machines (`SVM`)
    * A linear classifier that finds the maximal classifier between classes. In the anomaly detection case, we train a 1-class SVM, which fits all points onto a hypersphere. Anomaly detection is then performed by finding the distance to the center of the sphere; higher distances are anomolous.
* Generative Adversarial Networks (`GAN`) (experimental)
    * An architecture consisting of 2 portions: A generator model and a discriminator model. These two models are trained in parallel, and a combined anomaly score is computed based on a combination of the generator and discriminator losses. Implementation is based off of the [MAD-GAN paper](https://arxiv.org/abs/1901.04997).

## Execution

There are two ways to run and develop around the code.   
1) Run it from a jupyter notebook or 2) Run it in terminal yourself via Python3. 

### (1) Jupyter Notebook

Start the notebook. From the project root: `jupyter notebook .`   
This will open a browser window with the notebook environment filesystem.  
Similar to the provided demo, you open a new notebook file, and import the modules you need:
 - `main_train` for loading, training, and saving models
 - `main_eval` for running comparative evaluation experiments
 - `data_loader` for loading data
 - `detector` for interacting with the inner Keras API more directly

### (2) In terminal

Take a look at the files `main_train.py` and `main_eval.py`.  
In addition to being imported for the needed functions, they are also runnable as standalone demos.

To train new detection models: `python3 main_train.py 'model_type' 'dataset'`  
To plot, evaluate and compare detection models: `python3 eval_main.py 'model_type' 'dataset'`  

For example: `python3 main_train.py AE BATADAL`

The following model types are supported: `'AE', 'CNN', 'LSTM'`.  
The following datasets are supported: `'BATADAL', 'WADI', 'SWAT', 'SWATv2', 'gas-pipeline', 'HAI'`.

## Using Detectors in Other Applications
Detectors that are trained and tested by this framework can be used by other applications independently of this framework.

Internally, all models trained with this framework conform to the Keras API. The detector can be embedded and used within any python application by importing the `detector` module and calling the `train`, `predict`, `detect`, and `reconstruction_errors` functions accordingly. 

See the method stubs and generic functions in the abstract class definition of [detector.py]( https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/src/master/detector/detector.py).

Code for detectors that are packaged with this framework can be found in the [./detector](https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/src/master/detector/) directory.

## Training

### Model and Training Config (new)

> As of Version 0.1.2, we've switched to using command line arguments instead of JSON objects.

We use the argparse library. 
Running the script with `python3 main_train.py --help` will display all the mandatory and required arguments to the script.
Each configuration value, defaults and documentation are stored in the `utils.py` file.

#### Model parameters

| Name      | Description | Default |
| --- | --- | --- |
| '--ae_model_params_layers'   | The number of hidden layers in the autoencoder, between the input layer and the center embedding. | 5 |
| '--ae_model_params_cf'   | The autoencoder compression factor. The embedding layer will be of size (input size / cf). | 2.5 |
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

#### Detection hyperparameters

| Name      | Description | Default |
| --- | --- | --- |
| '--detect_params_percentile' | An array of percentile values optimize over. | Array of values from 0.95 to 0.99995 |
| '--detect_params_windows' | An array of window values optimize over. | Array of values from 1 to 100 |
| '--detect_params_metrics' | An array of metrics to use when optimizing. | ['F1'] |
| '--detect_params_plots' | If specified, save plots for each hyperparameter configuration. | False |
| '--detect_params_save_npy' | If specified, save npy files storing metric values for each hyperparameter configuration. | False |

#### Other 

| Name      | Description | Default |
| --- | --- | --- |
| '--gpus' | Which GPUS to use during training and evaluation? This should be specified as a GPU index value, as it is passed to the environment variable `CUDA_VISIBLE_DEVICES`. | None |
| '--run_name' | If provided, stores all models, plots and npy files in the associated `run_name` directory. Note: If saving plots, npys, or models, each of `models/run_name`, `npys/run_name`, and `plots/run_name` directories must exist. | None |

### Model and Training Config (old)

A different configuration was used pre-version 0.1.2 (Nov 2020), and the details can be found here: [Version 0.1.1 configuration](https://cement.andrew.cmu.edu/clementf/ics-ml-evasion-tool/src/master/reference/v0.1.1-config.md)

### Training API

The notebook documentation will give a better picture on how the workflow looks like, but `main_train.py` supports 4 main functions:

**train_model(model_type, config, Xtrain, Xval, dataset_name)**  
Trains and returns a detector object based on the provided train and validation data. Since no attack labels are given, this is an unsupervised regression task.

**explore_errors(event_detector, model_type, config, Xtrain, Xval, Xtest, Ytest, sensor_cols)**  
Given a detector, explores some of the train/test reconstruction error discrepancies. This is highly relevant for understanding the SWAT/WADI datasets.  
Generates several plots for feature-by-feature errors and potential detection issues.  
See also the following notebooks: `explore-wadi-errors.ipynb` and `explore-swat-errors.ipynb`

**hyperparameter_search(event_detector, model_type, config, Xval, Xtest, Ytest, test_split=0.7, save=False)**  
Perform a grid search of model hyperparameters, such as the detection threshold (theta) and the detection length (window)
If save=True, writes the theta values to a pickle file, for use later when evaluating models.
`test_split` specifies the proportion of the test set to be used for testing; the rest is used in detection hyperparameter tuning.

**save_model(event_detector, config)**  
Saves the model in pickle format. Needed for downstream scripts, such as evaluation, adversarial attacks, and explanations.

## Evaluation

### Evaluation Config

As shown in the notebook and script, configuration is passed to the training as a JSON object.

Top level parameters that need to be provided:
- 'name': A string that provides a naming identifier when saving and restoring models, plots and parameters.
- 'eval': A dictionary that configures the evaluation process

#### Evaluation parameters

The 'eval' configuration expects an array of dictionaries. Each element is a dictionary containing `window` and `percentile` and specify which versions of the trained model are used and compared against.  
The provided name must match a previous version of the model that has been trained and saved.

Here is an example that compares three of the saved models:
````
config = {
    'name': 'experiment1',
    'eval': [
        {'window': 1, 'percentile': 0.95},
        {'window': 5, 'percentile': 0.95},
        {'window': 10, 'percentile': 0.95}
    ]
}
````

### Evaluation API

We support two functions for evaluation:

**eval_test(model_type, dataset_name, model_name, percentile, window, plot=False)**  
Loads the model given the model type and name, loads the test data under the given dataset name, and performs an evaluation of the test data. Returns the test data `Ytest` and its corresponding prediction `Ytesthat`.

**eval_demo(model_type, dataset_name, config)**  
Given an instance of the above config and the name of a dataset, performs the evaluation, compares them, and produces a plot, saved under `{model_name}-{dataset_name}-compare.pdf`.

## Explanations

The mainline explanations code is stored in `main_explain.py`.

By running this code with a specified model (similar to how models are defined in training), this script will:
1. Find all true positive predicted regions in the test dataset
2. Capture the first N (default 120) samples in each attack
3. Executes an explanation method on all captured samples with the specified explanation methods in config.
4. Stores the values in a .pkl file, to be used later for scoring and visualization.

For example: `python3 main_explain.py CNN SWAT-CLEAN --explain_params_methods SM SG IG EG`

If the original model was trained under a specific `run_name` or with any `model_params`, they will need to be specified in this file as well, to ensure the same model is explained.

#### Explanation config

| Name      | Description | Default |
| --- | --- | --- |
| '--explain_params_methods' | Provides a list of explanation methods to use. Choices are saliency maps (SM), SmoothGrad (SG), integrated gradients (IG), and expected gradients (EG) | SM, SG |
| '--explain_params_use_top_feat' | If specified, use the top feature MSE as the gradient-based function, as opposed to the entire MSE.  | False |
| '--explain_params_threshold' | Percentile detection threshold for selecting candidates in true positives for explanation. Specifying this value implies a window value of 1. Setting a threshold of 0 (default) chooses the stored optimal detection hyperparameters. | 0 (optimal) |

### Explanation API

For each explanation method, a separate module is required in the `explainers/` directory. 
The required API is:

**setup_explainer(model, Xtrain, output_feature, sensor_cols)**  
Prepares the explanation method. For most techniques, this may involve training a surrogate model or simply setting up the required gradient functions.
model: Inner ML model to be explained
Xtrain: If needed, training data for surrogate model.
output_feature: Target output value to explain. Assumes classification output or a `top_feat` explanation.
sensor_cols: names of features, used for some explanation APIs.
    
**explain(Xexplain, *explain_params)**  
The core explanation method. Xexplain is a set of data, with the same shape as the model input, to be explained.
Each explanation method has various parameters that tweak how the explanation is done, so the rest of the parameters are 
generally defined.

### Post Explanations

A helper file provided is in `main_process_explain_gifs.py`, which will produce plots for each timestep, creating a time-series explanation over the course of the attack. 

For example, `python3 main_process_explain_gifs.py CNN SWAT-CLEAN explain-pkl/ 1` will create a set of .png files based on the explanations for attack 1 in SWAT.  

A helper script `bulk_gifs.sh` allows this process to be done in bulk, and will convert several attacks into GIFs in a loop.

## Adversarial Attacks

Code is still under development.
