# James Allingham
# April 2018
# MeanImpute.py
# Imputation by replacing missing values with the mean for the collumn

import numpy as np
import warnings

from Model import Model

class MeanImpute(Model):

    def __init__(self, data, verbose=None):
        Model.__init__(self, data, verbose=verbose)

        self.expected_X = self.X
        means = np.nanmean(self.X, axis=0)
        missing_locs = np.isnan(self.X)
        self.expected_X[missing_locs] = means[np.where(missing_locs)[1]]

        # self.ll = np.log(np.sum(np.stack([means]*self.N, axis=0) == self.expected_X))
        ll = 0
        for i in range(self.N):
            ll += np.log(np.array_equal(means, self.expected_X[i, :]))
        self.ll = ll/self.N


    def sample(self, n):
        warnings.warn("Cannot sample from a mean imputation. Returning the means.")
        return self.expected_X
