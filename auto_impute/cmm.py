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

    def __init__(self, data, num_components, verbose=None, α0=None, a0=None, map_est=True):
        Model.__init__(self, data, verbose=verbose)
        self.num_components = num_components

        self.map_est = map_est

        # hyper-parameters
        if α0 is not None: # pseudo counts for the mixing proportions
            self.α0 = α0
        else:
            self.α0 = 1

        if a0 is not None: # pseudo counts for the catagoricals
            self.a0 = a0
        else:
            self.a0 = 1

        # get list of unique values in each column
        self.unique_vals = [np.unique(self.X[:, d].compressed()) for d in range(self.num_features)]

        # create a dictionary from value to one-hot encoding of which unique value it is
        self.one_hot_lookups = [{val: encode_1_hot(i, len(self.unique_vals[d])) 
            for i, val in enumerate(self.unique_vals[d])}
                for d in range(self.num_features)]

        # randomise the initial parameter values
        # categorical distributions
        self.ps = np.array([[np.random.dirichlet(np.ones(len(self.unique_vals[d]))) 
            for d in range(self.num_features)] for k in range(self.num_components)])
        # mixing proportions
        self.πs = np.random.dirichlet([1]*self.num_components)

        # randomise initial responsibilities
        self.rs = np.random.dirichlet(np.ones((self.N,))*10, self.num_components).T

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_ps, old_πs, old_rs, old_expected_X = self.ps.copy(), self.πs.copy(), self.rs.copy(), self.expected_X.copy()

            # E-step
            self._calc_rs()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll - best_ll < ϵ:
                self.ps, self.πs, self.rs, self.expected_X = old_ps, old_πs, old_rs, old_expected_X
                self.ll = best_ll
                break
            
            best_ll = self.ll
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

    # E-step
    def _calc_rs(self):
        rs = np.zeros(shape=(self.N, self.num_components))

        for n in range(self.N):

            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask
            
            o_locs, _, _, _, _, _ = get_locs_and_coords(mask_row)

            if (o_locs.size):
                for k in range(self.num_components):
                    tmp = 1
                    for d in o_locs:
                        tmp *= stats.multinomial.pmf(self.one_hot_lookups[d][x_row[d]], 1, self.ps[k, d])
                    rs[n, k] = self.πs[k]*tmp
            else:
                rs[n, :] = self.πs

        self.rs = rs/np.sum(rs, axis=1, keepdims=True)

    # M-step
    def _update_params(self):
        # recompute πs
        if self.map_est:
                Ns = np.sum(self.rs, axis=0)

                # update the priors
                αs = self.α0 + Ns
                self.πs = (αs - 1)/np.sum(αs - 1)
        else:
            self.πs = np.mean(self.rs, axis=0)


        # now recompute ps
        if self.map_est:
            # include pseudo counts
            ps = np.array([[np.array([self.a0]*self.unique_vals[d].size, dtype=np.float64) for d in range(self.num_features)] for k in range(self.num_components)])
        else:
            # start counts at 0
            ps = np.array([[np.zeros(shape=(self.unique_vals[d].size), dtype=np.float64) for d in range(self.num_features)] for k in range(self.num_components)])

        for k in range(self.num_components):
            for d in range(self.num_features):
                for n in range(self.N):
                    ps[k, d] += self.rs[n, k]*(self.one_hot_lookups[d][self.X.data[n, d]] if not self.X.mask[n, d] else self.ps[k, d])

                ps[k, d] /= np.sum(ps[k, d])

        self.ps = ps
        
    def _calc_ML_est(self): 
        self.expected_X = self.X.data.copy()
        
        for n in range(self.N):
            for d in range(self.num_features):

                if self.X.mask[n, d]:
                    # determine which cluster to use
                    k = np.argmax(self.rs[n, :])
                    # now the max probability gives us the class
                    class_idx = np.argmax(self.ps[k, d])

                    self.expected_X[n, d] = self.unique_vals[d][class_idx]

    def _calc_ll(self):
        lls = []

        for n in range(self.N):
            mask_row = self.X[n, :].mask
            if np.all(~mask_row): continue

            prob = 0
            for k in range(self.num_components):
                tmp = 1
                for d in range(self.num_features):
                    if self.X.mask[n, d]:
                        tmp *= stats.multinomial.pmf(self.one_hot_lookups[d][self.expected_X[n, d]] ,1, self.ps[k, d])

                prob += self.rs[n, k]*tmp
            lls.append(np.log(prob)) 

        self.ll = np.mean(lls)

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):

            for n in range(self.N):
                mask_row = self.X[n, :].mask
                # if there are no missing values then go to next iter
                if np.all(~mask_row): continue

                for d in range(self.num_features):

                    if self.X.mask[n, d]:
                        k = np.random.choice(self.num_components, p=self.rs[n, :]) # maybe  self.πs?
                        # sample from a categorical distro using these probs
                        class_idx = np.argmax(stats.multinomial.rvs(1, self.ps[k, d]))

                        sampled_Xs[i, n, d] = self.unique_vals[d][class_idx]

        return sampled_Xs
