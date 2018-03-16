# James Allingham
# March 2018
# Model.py
# Base class for all imputation models

import numpy as np

class Model(object):

    def __init__(self, data):
        """Creates the model object and fits the model to the data.
        """
        # normalise the data for numerical stability
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.X = (data - self.mean)/self.std

        self.num_examples = data.shape[0]
        self.num_features = data.shape[1]

        self.imputed_X = None

    def impute(self):
        """Returns the imputed data
        """
        return self.imputed_X*self.std + self.mean

    def __impute():
        """Helper function for imputing.
        """
        raise NotImplementedError 

    def log_likelihood():
        """Calculates the log likelihood of the repaired data given the model paramers.
        """
        raise NotImplementedError 
