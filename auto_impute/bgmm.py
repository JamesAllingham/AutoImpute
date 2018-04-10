# James Allingham
# April 2018
# bgmm.py
# Imputation using a Gaussian Mixture Model fitted using variational Bayes

from model import Model

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg

class BGMM(Model):

    def __init__(self, data, num_gaussians, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians

        # hyper-parameters
        self.α0 = np.array([1e-3]*self.num_gaussians)
        self.m0 = np.zeros(shape=(self.num_features,))
        self.β0 = 1
        self.W0 = np.eye(self.num_features)
        self.ν0 = self.num_features

        # updated params
        self.αk = self.α0
        self.mk = self.m0
        self.βk = self.β0
        self.Wk = self.W0
        self.νk = self.ν0

        self._calc_expectation()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        pass

    def _calc_expectation(self):
        pass
    
    def _calc_ll(self):
        pass

    def sample(self, n):
        pass
