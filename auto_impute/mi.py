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
        Model.__init__(self, data, verbose=verbose, normalise=False)

        self.expected_X = self.X.data
        means = ma.mean(self.X, axis=0)
        
        # if there are no observations in any column of X then use 0.0
        means[np.isnan(means)] = 0

        # replace all missing values with the mean of the collumn
        self.expected_X[self.X.mask] = means[np.where(self.X.mask)[1]]

        # determine the lls for all of the values
        for n in range(self.N):
            for d in range(self.D):
                self.lls[n, d] = np.log(stats.norm.pdf(self.expected_X[n, d], loc=means[d], scale=1e-1)) # adding a little leeway here

    def _sample(self, num_samples):
        if self.verbose: 
            print_err("Cannot sample from a mean imputation. Returning the means.")

        return np.stack([self.expected_X]*num_samples, axis=0)
