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

### Workflow 1b - GRU on SWaT Dataset

This workflow will walk you through training a GRU model on the SWaT dataset, as well as generating explanations on a singular attack.

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a GRU model on the SWaT dataset:
```sh
python main_train.py GRU SWAT --train_params_epochs 10
```
This will utilize a default configuration of two layers, a history length of 50, and 64 units per layer for the GRU model. 
See detailed explanations for main_train.py parameters [here](README.md#parameters).

Next, use the GRU model to make predictions on the SWaT test dataset, and save the corresponding MSES.
```sh
python save_model_mses.py GRU SWAT
```

Additionally, use the GRU model to make predictions on the SWaT test dataset, and save the corresponding detection points. 
The default detection threshold is set at the 99.95-th percentile validation error.
```sh
python save_detection_points.py --md GRU-SWAT-l2-hist50-units64-results
```

Run attribution methods for SWaT attack #1, using the scripts in the `explain-eval-attacks` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-attacks
python main_grad_explain_attacks.py GRU SWAT 1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_attacks.py GRU SWAT 1 --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_attacks.py GRU SWAT 1 --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-swat.sh` are provided for reference.

Finally, rank the attribution methods for SWaT attack #1: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties.py 1 --md GRU-SWAT-l2-hist50-units64-results
```

### Workflow 2b - GRU on TEP Dataset

We provide another example that evaluates attribution methods on our synthetic manipulations: this workflow is similar to workflow 1 but is performed on the TEP dataset. 
Because of differences in how features are internally represented between datasets, the workflow uses slightly modified scripts specifically for dealing with the TEP dataset. 
This will also generate explanations on a singular TEP attack.
The sample attack used for this workflow is provided in `tep-attacks/matlab/TEP_test_cons_p2s_s1.csv`, which is a constant, two-standard-deviation manipulation on the first TEP sensor.

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a GRU model on the TEP dataset:
```sh
python main_train.py GRU TEP --train_params_epochs 10
```

Next, use the GRU model to make predictions on the TEP manipulation and save the corresponding MSES.
```sh
python save_model_mses.py GRU TEP
```

Additionally, use the GRU model to make predictions on the TEP manipulation and save the corresponding detection points. 
```sh
python save_detection_points.py --md GRU-TEP-l2-hist50-units64-results
```

Run attribution methods for the TEP manipulation, using the scripts in the `explain-eval-manipulations` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-manipulations
python main_tep_grad_explain.py GRU TEP cons_p2s_s1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_manipulations.py GRU TEP --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_manipulations.py GRU TEP --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-tep.sh` are provided for reference.

Finally, rank the attribution methods for the TEP manipulation: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties_tep.py --md GRU-TEP-l2-hist50-kern3-units64-results
```

### Workflow 1c - LSTM on SWaT Dataset

This workflow will walk you through training a LSTM model on the SWaT dataset, as well as generating explanations on a singular attack.

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a LSTM model on the SWaT dataset:
```sh
python main_train.py LSTM SWAT --train_params_epochs 10
```
This will utilize a default configuration of two layers, a history length of 50, and 64 units per layer for the LSTM model. 
See detailed explanations for main_train.py parameters [here](README.md#parameters).

Next, use the LSTM model to make predictions on the SWaT test dataset, and save the corresponding MSES.
```sh
python save_model_mses.py LSTM SWAT
```

Additionally, use the LSTM model to make predictions on the SWaT test dataset, and save the corresponding detection points. 
The default detection threshold is set at the 99.95-th percentile validation error.
```sh
python save_detection_points.py --md LSTM-SWAT-l2-hist50-units64-results
```

Run attribution methods for SWaT attack #1, using the scripts in the `explain-eval-attacks` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-attacks
python main_grad_explain_attacks.py LSTM SWAT 1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_attacks.py LSTM SWAT 1 --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_attacks.py LSTM SWAT 1 --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-swat.sh` are provided for reference.

Finally, rank the attribution methods for SWaT attack #1: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties.py 1 --md LSTM-SWAT-l2-hist50-units64-results
```

### Workflow 2c - LSTM on TEP Dataset

We provide another example that evaluates attribution methods on our synthetic manipulations: this workflow is similar to workflow 1 but is performed on the TEP dataset. 
Because of differences in how features are internally represented between datasets, the workflow uses slightly modified scripts specifically for dealing with the TEP dataset. 
This will also generate explanations on a singular TEP attack.
The sample attack used for this workflow is provided in `tep-attacks/matlab/TEP_test_cons_p2s_s1.csv`, which is a constant, two-standard-deviation manipulation on the first TEP sensor.

First, create the needed directories that will be populated with metadata:
```sh
bash make_dirs.sh
```

Next, train a LSTM model on the TEP dataset:
```sh
python main_train.py LSTM TEP --train_params_epochs 10
```

Next, use the LSTM model to make predictions on the TEP manipulation and save the corresponding MSES.
```sh
python save_model_mses.py LSTM TEP
```

Additionally, use the LSTM model to make predictions on the TEP manipulation and save the corresponding detection points. 
```sh
python save_detection_points.py --md LSTM-TEP-l2-hist50-units64-results
```

Run attribution methods for the TEP manipulation, using the scripts in the `explain-eval-manipulations` directory. 
Saliency maps (SM), SHAP, and LEMNA can be executed as follows. 
Each script will collect all attribution scores for 150 timesteps.
```sh
cd explain-eval-manipulations
python main_tep_grad_explain.py LSTM TEP cons_p2s_s1 --explain_params_methods SM --run_name results --num_samples 150
python main_bbox_explain_manipulations.py LSTM TEP --explain_params_methods SHAP --run_name results --num_samples 150
python main_bbox_explain_manipulations.py LSTM TEP --explain_params_methods LEMNA --run_name results --num_samples 150
```

Bash scripts `expl-full-bbox.sh` and `expl-full-tep.sh` are provided for reference.

Finally, rank the attribution methods for the TEP manipulation: the four attribution methods (baseline MSE, SM, SHAP, LEMNA) will each be ranked and compared with our various timing strategies:
```sh
cd .. # Return to root directory
python main_feature_properties_tep.py --md LSTM-TEP-l2-hist50-kern3-units64-results
```

