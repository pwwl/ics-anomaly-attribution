"""

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

# classes
class SmoothGradHistoryExplainer(ICSExplainer):
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

        self.name = 'smooth_gradients_history'
        self.params = params
        self.inner = None
        self.explainer = None

    def setup_explainer(self, model, Ytrue, top_feat):
        """ Creates a wrapper around the given explanation method.      

            Attributes:
            model: Inner ML model to be explained
            Xtrain: Training data for model.
            output_feature: Target output value to explain. Assumes classification output.

        """
        self.inner = model

        # MSE of model output and reconstruction goal (last part of input)
        top_loss = (self.inner.output[:, top_feat] - Ytrue[:, top_feat])**2
        
        grads = K.gradients(
          top_loss,
          self.inner.input
          )[0]

        # Sets up the gradient function
        self.explainer = K.function([self.inner.input], [grads])

        return 

    def explain(self, Xexplain, baselines=None, n_steps = 50, multiply_by_input=True, **explain_params):
        """ Return an explanation for Xexplain.

            Attributes:
            Xexplain: Target to explain. Can be an nxd example.
        """

        # If vector given, clean input into a 1xd
        if np.ndim(Xexplain) == 1:
          Xexplain_clean = Xexplain.reshape(1, len(Xexplain))
        else:
          Xexplain_clean = Xexplain

        # Find baseline, use 0s as default
        if baselines is None:
          baselines = 0 * Xexplain_clean

        smooth_grad = np.zeros_like(Xexplain_clean)
        sigma = np.sqrt(0.1 * (np.max(Xexplain_clean) - np.min(Xexplain_clean)))

        for k in range(n_steps):
          noise = sigma * np.random.randn(*Xexplain_clean.shape)
          noised_value = Xexplain_clean + noise
          smooth_grad += self.explainer([noised_value])[0]

        if multiply_by_input:
            attributions = smooth_grad * (Xexplain_clean) * (1/n_steps)
        else:
            attributions = smooth_grad * (1/n_steps)

        return attributions

class SaliencyMapHistoryExplainer(ICSExplainer):
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

        self.name = 'saliency_map_history'
        self.params = params
        self.inner = None
        self.explainer = None

    def setup_explainer(self, model, Ytrue, top_feat):
        """ Creates a wrapper around the given explanation method.      

            Attributes:
            model: Inner ML model to be explained
            Xtrain: Training data for model.
            output_feature: Target output value to explain. Assumes classification output.

        """
        self.inner = model

        # MSE of model output and reconstruction goal (last part of input)
        top_loss = (self.inner.output[:, top_feat] - Ytrue[:, top_feat])**2
        
        grads = K.gradients(
          top_loss,
          self.inner.input
          )[0]

        # Sets up the gradient function
        self.explainer = K.function([self.inner.input], [grads])

        return 

    def explain(self, Xexplain, baselines=None, n_steps = 50, multiply_by_input=True, **explain_params):
        """ Return an explanation for Xexplain.

            Attributes:
            Xexplain: Target to explain. Can be an nxd example.
        """

        # If vector given, clean input into a 1xd
        if np.ndim(Xexplain) == 1:
          Xexplain_clean = Xexplain.reshape(1, len(Xexplain))
        else:
          Xexplain_clean = Xexplain

        single_grad = self.explainer([Xexplain_clean])[0]

        if multiply_by_input:
            attributions = single_grad * (Xexplain_clean)
        else:
            attributions = single_grad

        return attributions

if __name__ == "__main__":
    print("Not a main file.")
