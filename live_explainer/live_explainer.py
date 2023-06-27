"""

   Copyright 2022 Lujo Bauer, Clement Fung

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

# classes
class LiveExplainer(object):
    """ Live Explainer for ICS event detection models.

        Attributes:
        params: dictionary with parameters defining the model structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize AE Keras model. """

        # Default parameter values.
        params = {
            'verbose' : 0,
            'method' : 'Default'
            }

        for key, item in kwargs.items():
            params[key] = item

        self.name = 'Default'
        self.params = params
        self.inner = None

    def score_generate(self, Xexplain, **explain_params):
        """ Generate explanation scores for the given Xexplain.

            Xexplain: input to explanation method.
        """
        return NotImplementedError

    def get_name(self):
        """ Return explainer name
        
        """
        return self.name


if __name__ == "__main__":
    print("Not a main file.")
