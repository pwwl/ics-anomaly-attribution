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

# keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, BatchNormalization, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad

import json

# ### Autoencoder classes
# classes
class ICSDetector(object):
    """ Keras-based Detector for ICS event detection.

        Attributes:
        params: dictionary with parameters defining the model structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize AE Keras model. """

        # Default parameter values.
        params = {
            'verbose' : 0
            }

        for key,item in kwargs.items():
            params[key] = item

        self.params = params

    def create_model(self):
        """ Creates Keras model.
        """

        # retrieve params
        verbose = self.params['verbose'] # echo on screen

        # create model
        inner_model = Sequential()

        # print autoencoder specs
        if verbose > 0:
            print('Default model creation')

        self.inner = inner_model
        return inner_model

    def train(self, x, **train_params):
        """ Stub Method for training

            x: inputs (inputs == targets, AE are self-supervised ANN).
        """
        return NotImplementedError

    def detect(self, x, theta, window = 1, **keras_params):
        """ Stub Method for detection
        """
        return NotImplementedError

    def best_detect(self, x, **keras_params):
        """ Stub Method for detection, using the stored best theta and window values
        """
        return self.detect(x, theta=self.params['best_theta'], window=self.params['best_window'], **keras_params)

    def cached_detect(self, instance_errors, theta, window = 1):   
        """ Stub Method for detection, using precomputed errors (for efficiency)
        """
        return NotImplementedError

    def best_cached_detect(self, instance_errors):
        """ Stub Method for detection, using precomputed errors (for efficiency) and the stored best parameters
        """
        return self.cached_detect(instance_errors, theta=self.params['best_theta'], window=self.params['best_window'])

    def reconstruction_errors(self, x):
        """ Get reconstruction errors on input data
        """
        return NotImplementedError

    def predict(self, x, **test_params):
        """ Yields reconstruction error for all inputs,

            x: inputs.
        """
        return self.inner.predict(x, **test_params)

    def get_inner(self):
        """ Return the inner model
        """
        return self.inner

    def save_detection_params(self, best_theta, best_window):

        print(f'Storing best parameters as: theta={best_theta} window={best_window}')
        
        self.params['best_theta'] = best_theta
        self.params['best_window'] = best_window

    def save(self, filename):
        """ Save AEED modelself.

            AEED parameters saved in a .json, while Keras model is stored in .h5 .
        """
        # parameters
        with open(filename+'.json', 'w') as fp:
            json.dump(self.params, fp)
        # keras model
        self.inner.save(filename+'.h5')
        # echo
        print('Saved model parameters to {0}.\nKeras model saved to {1}'.format(filename+'.json', filename+'.h5'))

if __name__ == "__main__":
    print("Not a main file.")
