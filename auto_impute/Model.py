# James Allingham
# March 2018
# Model.py
# Base class for all imputation models

import numpy as np

class Model(object):

    def __init__(self, data, verbose=None):
        """Creates the model object and fits the model to the data.
        """
        # normalise the data for numerical stability
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)

        # self.X = (data - self.mean)/self.std
        self.X = data

        self.N = data.shape[0]
        self.num_features = data.shape[1]

        # check that the data is somewhat reasonable
        if self.N < 1: raise RuntimeError("Input data must have at least one example.") # consider adding specific exception classes for these
        if self.num_features < 1: raise RuntimeError("Input data must have at least one feature.")
        if np.any(np.isnan(np.nanmean(data, axis=0))): raise RuntimeError("Each feature must have at least one observed value.")

        self.expected_X = np.array([])
        self.ll = None

        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

    def impute(self):
        """Returns the imputed data
        """
        # return self.expected_X*self.std + self.mean
        return self.expected_X

    # def __calc_expectation(self):
    #     """Helper function for calculating the expectation of the missing data.
    #     """
    #     raise NotImplementedError 

    def log_likelihood(self):
        """Calculates the log likelihood of the repaired data given the model paramers.
        """
        return self.ll 

    # def __calc_ll(self):
    #     """Helper function for log likelihood.
    #     """
    #     raise NotImplementedError 

    def sample(self, n):
        """Samples from the density.
        """
        raise NotImplementedError
