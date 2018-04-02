# James Allingham
# April 2018
# mean_impute.py
# Imputation by replacing missing values with the mean for the collumn

from Model import Model

from numpy import np
import warnings

class MeanImpute(Model):

    def __init__(self, data, verbose=None):
        Model.__init__(self, data, verbose=verbose)

        means = np.nanmean(self.expected_X, axis=0)
        missing_locs = np.isnan(self.expected_X)
        self.expected_X[missing_locs] = means[missing_locs[1]]

        self.ll = np.log(np.sum(np.stack([means]*self.N, axis=0) == self.expected_X))

    def sample(self):
        warnings.warn("Cannot sample from a mean imputation. Returning the means.")
        return self.expected_X
