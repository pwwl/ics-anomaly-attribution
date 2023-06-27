"""

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

"""

# numpy stack
import numpy as np
import pdb

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import json
import tensorflow.keras.backend as K
import tensorflow as tf

from .explainer import ICSExplainer

class ExpectedGradientsMseHistoryExplainer(ICSExplainer):
    """ Keras-based ML-Explainer for ICS event detection models.

        Attributes:
        params: dictionary with parameters defining the explainer structure
    """
    def __init__(self, **kwargs):
        """ Class constructor """

        # Default parameter values.
        params = {
            'verbose' : False,
            }

        for key, item in kwargs.items():
            params[key] = item

        self.name = 'expected_gradients_mse_history'
        self.params = params
        self.inner = None
        self.explainer = None

    def setup_explainer(self, model, Ytrue):
        """ Creates a wrapper around the given explanation method.      

            Attributes:
            model: Inner ML model to be explained
            Xtrain: Training data for model.
            output_feature: Target output value to explain. Assumes classification output.

        """
        self.inner = model

        # MSE of model output and reconstruction goal (last part of input)
        loss = K.mean((self.inner.output - Ytrue)**2)
        grads = K.gradients(
          loss,
          self.inner.input
          )[0]

        # Sets up the gradient function
        self.explainer = K.function([self.inner.input], [grads])

        return 

    def explain(self, Xexplain, baselines, n_steps = 200, **explain_params):
        """ Return an explanation for Xexplain.

            Attributes:
            Xexplain: Target to explain. Can be an nxd example.
            baselines: Set of (n x h x d) training inputs used to sample expectation from. This is mandatory for EG.
        """

        # If vector given, clean input into a 1xd
        if np.ndim(Xexplain) == 1:
          Xexplain_clean = Xexplain.reshape(1, len(Xexplain))
        else:
          Xexplain_clean = Xexplain

        expected_grad = np.zeros_like(Xexplain_clean)

        # EG: Sample a training input, and a [0,1] interpolation value
        for k in range(n_steps):

          # Sample input from given baselines
          sample_idx = np.random.randint(len(baselines))
          sample_baseline = baselines[sample_idx]

          # Sample [0,1] value 
          sample_interp = np.random.uniform()

          step_value = sample_baseline + sample_interp * (Xexplain_clean - sample_baseline)
          expected_grad += self.explainer([step_value])[0]

        attributions = expected_grad * Xexplain_clean * (1/n_steps)
        return attributions

if __name__ == "__main__":
    print("Not a main file.")
