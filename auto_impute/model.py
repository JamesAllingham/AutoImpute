# James Allingham
# March 2018
# Model.py
# Base class for all imputation models

import sys

import numpy as np
import numpy.ma as ma

from utilities import print_err

class Model(object):

    def __init__(self, data, verbose=None):
        """Creates the model object and fits the model to the data.
        """
        self.N = data.shape[0]
        self.D = data.shape[1]

        self.X = data

        # check that the data is somewhat reasonable
        if self.N < 1: 
            print_err("Input data must have at least one example.")
            sys.exit(1)

        if self.D < 1:
            print_err("Input data must have at least one feature.")
            sys.exit(1)

        self.expected_X = np.array([])
        self.lls = np.zeros_like(self.X.data)

        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

    def ml_imputation(self):
        """Returns the imputed data
        """
        return self.expected_X

    def log_likelihood(self, complete=False, observed=False, return_individual=False, return_mean=False):
        """Calculates the log likelihood of the repaired data given the model paramers.
        """
        lls = self.lls[~self.X.mask] if observed else self.lls[self.X.mask] if not complete else self.lls

        if return_individual:
            return lls

        if return_mean:
            return np.mean(lls)

        return np.sum(lls)

    def sample(self, num_samples):
        """Samples from the density.
        """
        return self._sample(num_samples)

    def _sample(self, num_samples):
        raise NotImplementedError
