# James Allingham
# April 2018
# mi.py
# Imputation by replacing missing values with the mean for the collumn

from model import Model
from utilities import get_locs_and_coords

import numpy as np
import numpy.ma as ma
from scipy import stats
import warnings

class MeanImpute(Model):

    def __init__(self, data, verbose=None):
        Model.__init__(self, data, verbose=verbose)

        self.expected_X = self.X.data
        means = ma.mean(self.X, axis=0)
        missing_locs = self.X.mask
        self.expected_X[missing_locs] = means[np.where(missing_locs)[1]]

        lls = []
        for n in range(self.N):
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            _, m_locs, _, _, _, _ = get_locs_and_coords(mask_row)

            lls.append(np.log(stats.multivariate_normal.pdf(self.expected_X[n, m_locs], mean=means[m_locs], cov=np.ones(m_locs.size)*1e-3)))
        self.ll = np.mean(lls)


    def _sample(self, num_samples):
        warnings.warn("Cannot sample from a mean imputation. Returning the means.")
        return np.stack([self.expected_X]*num_samples, axis=0)
