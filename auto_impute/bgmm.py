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
        self.α0 = 1e-3
        self.m0 = np.zeros(shape=(self.num_features,))
        self.β0 = 1
        self.W0 = np.eye(self.num_features)
        self.ν0 = self.num_features

        self.Xs = np.array([])

        self.__calc_expectation()
        self.__calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        pass

    def __calc_expectation(self):
        pass
    
    def __calc_ll(self):
        pass

    def sample(self, n):
        pass
