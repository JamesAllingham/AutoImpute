# James Allingham
# March 2018
# SingleGaussianEM.py
# Imputation using a single Gaussian distribution fitted using the EM algorithm

import Model

import numpy as np
from scipy import stats
from scipy import linalg

class SingleGaussian(Model.Model):

    def __init__(self, data, ϵ=None, max_iters=None, verbose=None):
        Model.Model.__init__(self, data, ϵ=ϵ, max_iters=max_iters, verbose=verbose)        
        # self.μ = np.random.rand(self.num_features)
        # self.Σ = np.random.rand(self.num_features, self.num_features)
        self.μ = np.nanmean(self.X, axis=0)
        self.Σ = np.nansum([np.outer(self.X[i,:] - self.μ, self.X[i,:] - self.μ) for i in range(self.N)], axis=0)

        # fit the model to the data
        best_ll = -np.inf

        if (self.verbose): print("Fitting model:")
        for i in range(self.max_iters):
            old_μ, old_Σ, old_expected_X = self.μ.copy(), self.Σ.copy(), self.expected_X.copy()

            if i == 0:
                # using the current parameters, estimate the values of the missing data:
                # impute by taking the mean of the conditional distro
                self.__calc_expectation()

            # now re-estimate μ and Σ (M-step)
            self.μ = np.mean(self.expected_X, axis = 0)
            self.Σ = np.zeros_like(self.Σ)
            for j in range(self.N):
                diff = self.expected_X[j,:] - self.μ
                self.Σ += np.outer(diff, diff.T)
            self.Σ = self.Σ/self.N 

            # regularisation term ensuring that the cov matrix is always pos def
            self.Σ += np.eye(self.num_features)*1e-3
            
            # using the current parameters, estimate the values of the missing data (E-step)
            # impute by taking the mean of the conditional distro
            self.__calc_expectation()

            # if the log likelihood stops improving then stop iterating
            self.__log_likelihood()
            if (self.ll < best_ll or self.ll - best_ll < self.ϵ):
                self.μ, self.Σ, self.expected_X = old_μ, old_Σ, old_expected_X
                self.ll = best_ll
                break
            
            best_ll = self.ll

            if (self.verbose): print("Iter: %s\t\tLL: %f" % (i, self.ll))
            
    def __calc_expectation(self):
        Λ = linalg.inv(self.Σ)
        expected_X = self.X.copy()
        for i in range(self.N):
            X_row = expected_X[i,:]
            # if there are no missing values then go to next iter
            if np.all(~np.isnan(X_row)): continue

            # figure out which values are missing
            o_locs = np.where(~np.isnan(X_row))[0]
            m_locs = np.where(np.isnan(X_row))[0]
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

            # calculate the mean of m|o
            μmo = self.μ[m_locs] 
            if (len(o_locs)): # if there are any observations
                # get the subsets of the precision matrices
                Λmm = Λ[mm_coords].reshape(len(m_locs), len(m_locs))
                Λmo = Λ[mo_coords].reshape(len(m_locs), len(o_locs))
                μmo -= linalg.inv(Λmm) @ Λmo @ (X_row[o_locs] - self.μ[o_locs])

            expected_X[i,:][m_locs] = μmo
        self.expected_X = expected_X

    def __log_likelihood(self):
        ll = 0
        for i in range(self.N):
                ll += stats.multivariate_normal.pdf(self.expected_X[i,:], mean=self.μ, cov=self.Σ)
        self.ll = np.log(ll)

    # def sample(self, n):
    #     sampled_Xs = np.stack([])
