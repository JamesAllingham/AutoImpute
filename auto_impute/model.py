# James Allingham
# March 2018
# Model.py
# Base class for all imputation models

import numpy as np
import numpy.ma as ma

class Model(object):

    def __init__(self, data, verbose=None):
        """Creates the model object and fits the model to the data.
        """
        # normalise the data for numerical stability
        self.mean = ma.mean(data, axis=0)
        self.std = ma.std(data, axis=0)
        # self.mean = 0
        # self.std = 1

        self.X = (data - self.mean)/self.std

        self.N = data.shape[0]
        self.num_features = data.shape[1]

        # check that the data is somewhat reasonable
        if self.N < 1: raise RuntimeError("Input data must have at least one example.") # consider adding specific exception classes for these
        if self.num_features < 1: raise RuntimeError("Input data must have at least one feature.")
        if np.any(np.all(data.mask, axis=0)): raise RuntimeError("Each feature must have at least one observed value.")

        self.expected_X = np.array([])
        self.ll = None

        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

    def impute(self):
        """Returns the imputed data
        """
        return self.expected_X*self.std + self.mean

    def log_likelihood(self):
        """Calculates the log likelihood of the repaired data given the model paramers.
        """
        return self.ll

    def sample(self, num_samples):
        """Samples from the density.
        """
        raise NotImplementedError
