# James Allingham
# April 2018
# mmm.py
# Imputation using a Mixed Mixture Model (consisiting of categorical and gaussian components) fitted using the EM algorithm
# Based on Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

# NOTE: scipy.stats.multinomial sometimes eroneously gives nan for Scipy versions lower than 1.10.0

from model import Model
from utilities import get_locs_and_coords, encode_1_hot, regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from sklearn.cluster import KMeans

class MMM(Model):

    def __init__(self, data, num_components, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        
        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μs, old_Σs, old_expected_X, old_Xs, old_rs = self.μs.copy(), self.Σs.copy(), self.expected_X.copy(), self.Xs.copy(), self.rs.copy()

            # E-step
            self._calc_rs()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.μs, self.Σs, self.expected_X, self.Xs, self.rs = old_μs, old_Σs, old_expected_X, old_Xs, old_rs
                self.ll = best_ll
                break
            
            best_ll = self.ll
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

    # E-step
    def _calc_rs(self):
        pass

    # M-step
    def _update_params(self):
        pass

    def _calc_ML_est(self):
        pass

    def _calc_ll(self):
        pass

    def sample(self, num_samples):
        pass
        