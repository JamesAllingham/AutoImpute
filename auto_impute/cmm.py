# James Allingham
# April 2018
# cmm.py
# Imputation using a Categorical Mixture Model fitted using the EM algorithm
# Based on Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

# NOTE: scipy.stats.multinomial sometimes eroneously gives nan for Scipy versions lower than 1.10.0

from model import Model
from utilities import get_locs_and_coords, encode_1_hot

import numpy as np
from scipy import stats

class CMM(Model):

    def __init__(self, data, num_gaussians, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians

        # get list of unique values in each column
        self.unique_vals = [np.unique(data[:, d].compressed()) for d in range(self.num_features)]

        # create a dictionary from value to one-hot encoding of which unique value it is
        self.one_hot_lookups = [{val: encode_1_hot(i, len(self.unique_vals[d])) 
            for i, val in enumerate(self.unique_vals[d])}
                for d in range(self.num_features)]

        # randomise the initial parameter values for the categorical distributions
        # self.rs = np.random.rand(self.N, self.num_gaussians)
        # self.rs = self.rs/np.sum(self.rs, axis=1, keepdims=True)
        self.rs = np.random.dirichlet(np.ones((self.N,)), self.num_gaussians).T

        self.ps = [np.random.dirichlet(np.ones(len(self.unique_vals[d])), self.num_gaussians).T for d in range(self.num_features)]

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_ps, old_rs = self.ps.copy(), self.rs.copy()

            # E-step
            self._calc_rs()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.ps, self.rs = old_ps, old_rs
                self.ll = best_ll
                break
            
            best_ll = self.ll
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

    # E-step
    def _calc_rs(self):
        rs = np.zeros(shape=(self.N, self.num_gaussians))

        for n in range(self.N):

            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask
            
            o_locs, _, _, _, _, _ = get_locs_and_coords(mask_row)

            if (o_locs.size):
                for k in range(self.num_gaussians):
                    tmp = 1
                    for d in o_locs:
                        tmp *= stats.multinomial.pmf(self.one_hot_lookups[d][x_row[d]], 1, self.ps[d][:, k])
                    rs[n, k] = tmp
            else:
                rs[n, :] = np.mean(self.rs, axis=0)

        self.rs = rs/np.sum(rs, axis=1, keepdims=True)

    # M-step
    def _update_params(self):
        # ps = np.zeros_like(self.ps)
        ps = [np.zeros(shape=(self.unique_vals[d].size, self.num_gaussians)) for d in range(self.num_features)]

        for k in range(self.num_gaussians):
            for d in range(self.num_features):
                tmp = 0
                for n in range(self.N):
                        tmp += self.rs[n, k]*(self.one_hot_lookups[d][self.X.data[n, d]] if not self.X.mask[n, d] else self.ps[d][:, k])

                tmp /= np.sum(self.rs[:, k])
                ps[d][:, k] = tmp

        self.ps = ps
        
    def _calc_ML_est(self): 
        self.expected_X = self.X.data.copy()
        
        for n in range(self.N):
            for d in range(self.num_features):

                if self.X.mask[n, d]:
                    # figure out the probabilities for each class based on the mixture responsibilites
                    p = np.sum(self.rs[n, :]*self.ps[d][:, :], axis=1)

                    # now the max probability gives us the class
                    class_idx = np.argmax(p)

                    self.expected_X[n, d] = self.unique_vals[d][class_idx]

    def _calc_ll(self):
        lls = []

        for n in range(self.N):
            mask_row = self.X[n, :].mask
            if np.all(~mask_row): continue

            prob = 0
            for k in range(self.num_gaussians):
                tmp = 1
                for d in range(self.num_features):
                    if self.X.mask[n, d]:
                        tmp *= self.rs[n, k]*stats.multinomial.pmf(self.one_hot_lookups[d][self.expected_X[n, d]] ,1, self.ps[d][:, k])

                prob += tmp
            lls.append(np.log(prob)) 

        self.ll = np.mean(lls)

    def sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):

            for n in range(self.N):
                mask_row = self.X[n, :].mask
                # if there are no missing values then go to next iter
                if np.all(~mask_row): continue

                for d in range(self.num_features):

                    if self.X.mask[n, d]:
                        # figure out the probabilities for each class based on the mixture responsibilites
                        p = np.sum(self.rs[n, :]*self.ps[:, d], axis=0)
                        # sample from a categorical distro using these probs
                        class_idx = np.argmax(stats.multinomial.rvs(1, p))

                        sampled_Xs[i, n, d] = self.unique_vals[d][class_idx]

        return sampled_Xs
