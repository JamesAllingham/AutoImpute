# James Allingham
# March 2018
# SingleGaussianEM.py
# Imputation using a single Gaussian distribution fitted using the EM algorithm

from Model import Model

import numpy as np
from scipy import stats
from scipy import linalg

class SingleGaussian(Model):

    def __init__(self, data, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.μ = np.nanmean(self.X, axis=0)
        self.Σ = np.nanmean([np.outer(self.X[i, :] - self.μ, self.X[i, :] - self.μ) for i in range(self.N)], axis=0)

        self.__calc_expectation()
        self.__calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        # fit the model to the data
        best_ll = self.ll

        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μ, old_Σ, old_expected_X = self.μ.copy(), self.Σ.copy(), self.expected_X.copy()

            # now re-estimate μ and Σ (M-step)
            self.μ = np.mean(self.expected_X, axis=0)
            self.Σ = np.zeros_like(self.Σ)
            for j in range(self.N):
                diff = self.expected_X[j, :] - self.μ
                self.Σ += np.outer(diff, diff.T)
            self.Σ = self.Σ/self.N 

            # regularisation term ensuring that the cov matrix is always pos def
            self.Σ += np.eye(self.num_features)*1e-3
            
            # using the current parameters, estimate the values of the missing data (E-step)
            # impute by taking the mean of the conditional distro
            self.__calc_expectation()

            # if the log likelihood stops improving then stop iterating
            self.__calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.μ, self.Σ, self.expected_X = old_μ, old_Σ, old_expected_X
                self.ll = best_ll
                break
            
            best_ll = self.ll

            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))
            
    def __calc_expectation(self):
        expected_X = self.X.copy()
        for i in range(self.N):
            x_row = expected_X[i, :]
            # if there are no missing values then go to next iter
            if np.all(~np.isnan(x_row)): continue

            # figure out which values are missing
            o_locs = np.where(~np.isnan(x_row))[0]
            m_locs = np.where(np.isnan(x_row))[0]
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))

            # calculate the mean of m|o
            μmo = self.μ[m_locs] 
            if o_locs.size: # if there are any observations
                # get the subsets of the precision matrices
                Σoo = self.Σ[oo_coords].reshape(len(o_locs), len(o_locs))
                Σmo = self.Σ[mo_coords].reshape(len(m_locs), len(o_locs))
                μmo += Σmo @ linalg.inv(Σoo) @ (x_row[o_locs] - self.μ[o_locs])

            expected_X[i, :][m_locs] = μmo
        self.expected_X = expected_X

    def __calc_ll(self):
        ll = 0
        for i in range(self.N):
            ll += np.log(stats.multivariate_normal.pdf(self.expected_X[i, :], mean=self.μ, cov=self.Σ))
        self.ll = ll/self.N

    def sample(self, n):
        sampled_Xs = np.stack([self.X.copy()]*n, axis=0)

        for i in range(self.N):
            # figure out the conditional distribution for the missing data given the observed data
            x_row = self.X[i, :]
            # if there are no missing values then go to next iter
            if np.all(~np.isnan(x_row)): continue

            # figure out which values are missing
            o_locs = np.where(~np.isnan(x_row))[0]
            m_locs = np.where(np.isnan(x_row))[0]
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

            for j in range(n):
                μmo = self.μ[m_locs]

                if o_locs.size:
                    Σoo = self.Σ[oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σ[mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μ[o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σ[mm_coords].reshape(len(m_locs), len(m_locs))

                sampled_Xs[j, i, m_locs] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

        return sampled_Xs
