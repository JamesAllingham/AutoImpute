# John Doe
# March 2018
# Model.py
# Base class for all imputation models

import sys

import numpy as np
import numpy.ma as ma

from utilities import print_err

class Model(object):

    def __init__(self, data, verbose=None):
        """Creates the model object.

        Args:
            data: The dataset with missing data as a numpy masked array.
            verbose: bool, indicating whether or not information should be written to std_err.

        Returns:
            The model.
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
        """Use the mode of the posterior distribution to impute the missing data.

        Args:
            None.

        Returns:
            A numpy array containing the imputed dataset.
        """
        return self.expected_X

    def log_likelihood(self, complete=False, observed=False, return_individual=False, return_mean=False):
        """Calculates the log likelihood of the repaired data given the model paramers.

        Args:
            complete: bool, if True then LLs for both the missing and non-missing data is returned, if False then only the missing data LLs are returned.
            observed: bool, if True, and complete is False, then only the observed LLs are returned, ignored if complete is True.
            return_individual: bool, if True the individual LLs are returned, if False the sum of the LLs is returned.
            return_mean: bool, if True and return individual is false, then the mean of the LLs is returned, ignored if return_individual is True.

        Returns:
            numpy array of individual, average or sum of complete, observed, or missing LLs depending on the paramters above.
        """
        lls = self.lls[~self.X.mask] if observed else self.lls[self.X.mask] if not complete else self.lls

        if return_individual:
            return lls

        if return_mean:
            return np.mean(lls)

        return np.sum(lls)

    def sample(self, num_samples):
        """Performs multiple imputation by sampling from the posterior distribution.

        Args:
            num_smaples: The integer number of datasets to sample from the posterior.

        Returns:
            num_samples imputed datasets.
        """
        return self._sample(num_samples)

    def _sample(self, num_samples):
        """Sampling helper function

        Args:
            num_smaples: The integer number of datasets to sample from the posterior.

        Returns:
            num_samples imputed datasets.
        """
        raise NotImplementedError
