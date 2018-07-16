# James Allingham
# April 2018
# mi.py
# Imputation by replacing missing values with the mean for the collumn

from model import Model
from utilities import print_err

import numpy as np
import numpy.ma as ma
from scipy import stats

class MeanImpute(Model):

    def __init__(self, data, verbose=None):
        """Creates the model object.

        Args:
            verbose: bool, indicating whether or not information should be written to std_err.

        Returns:
            The model.
        """
        Model.__init__(self, data, verbose=verbose)

        self.expected_X = self.X.data.copy()
        self.μ = ma.mean(self.X, axis=0)
        
        # if there are no observations in any column of X then use 0.0
        self.μ[np.isnan(self.μ)] = 0

        # replace all missing values with the mean of the collumn
        self.expected_X[self.X.mask] = self.μ[np.where(self.X.mask)[1]]

        # determine the lls for all of the values
        for n in range(self.N):
            for d in range(self.D):
                self.lls[n, d] = np.log(stats.norm.pdf(self.expected_X[n, d], loc=self.μ[d], scale=1e-1)) # adding a little leeway here

    def _sample(self, num_samples):
        """Sampling helper function.

        Note that mean imputation can't sample so this returns num_samples copies of the ML imputation.

        Args:
            num_smaples: The integer number of datasets to sample from the posterior.

        Returns:
            num_samples imputed datasets.
        """
        if self.verbose: 
            print_err("Cannot sample from a mean imputation. Returning the means.")

        return np.stack([self.expected_X]*num_samples, axis=0)

    def test_ll(self, test_data):
        """LL for unseen test data.

        Args:
            test_data: a numpy array to calculate the LL for.

        Returns:
            A numpy array the same size as test_data with containing the LLs for each entry in test_data
        """
        N, D = test_data.shape
        if not D == self.D: 
            print_err("Dimmensionality of test data (%s) not equal to dimmensionality of training data (%s)." % (D, self.D))

        lls = np.zeros_like(self.lls)
        for n in range(self.N):
            for d in range(self.D):
                lls[n, d] = np.log(stats.norm.pdf(test_data[n, d], loc=self.μ[d], scale=1e-1))

        return lls
