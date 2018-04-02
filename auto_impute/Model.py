# James Allingham
# March 2018
# Model.py
# Base class for all imputation models

import numpy as np

class Model(object):

    def __init__(self, data, ϵ=None, max_iters=None, verbose=None):
        """Creates the model object and fits the model to the data.
        """
        # normalise the data for numerical stability
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        # self.X = (data - self.mean)/self.std
        self.X = data

        self.N = data.shape[0]
        self.num_features = data.shape[1]

        self.expected_X = np.array([])
        self.ll = None

        # refused bequest
        if (ϵ is None):
            self.ϵ = 1e-1
        else:
            self.ϵ = ϵ

        # refused bequest
        if (max_iters is None):
            self.max_iters = 100
        else:
            self.max_iters = max_iters

        if (verbose is None):
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

    # def __log_likelihood(self):
    #     """Helper function for log likelihood.
    #     """
    #     raise NotImplementedError 

    def sample(self, n):
        """Samples from the density.
        """
        raise NotImplementedError
