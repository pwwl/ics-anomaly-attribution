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
class MSEExplainer(object):
    """ Live Explainer for ICS event detection models.

        Attributes:
        params: dictionary with parameters defining the model structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor """

        # Default parameter values.
        params = {
            'verbose' : 0,
            'method' : 'MSEExplainer'
            }

        for key, item in kwargs.items():
            params[key] = item

        self.name = 'MSEExplainer'
        self.params = params

        # Supply the given event_detector
        self.inner_detector = params['detector']

    def score_generate(self, Xtest, index_selection=None, **explain_params):
        """ Using MSE argmaxing, generate explanation scores for the given Xtest.

            Xtest: input to explanation method.
        """
        
        full_test_errors = self.inner_detector.reconstruction_errors(Xtest, batches=True)
        relevant_test_errors = full_test_errors[index_selection]

        return relevant_test_errors

    def get_name(self):
        """ Return explainer name
        
        """
        return self.name


if __name__ == "__main__":
    print("Not a main file.")
