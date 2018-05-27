# James Allingham
# March 2018
# sg.py
# Imputation using a single Gaussian distribution fitted using the EM algorithm

from model import Model
from utilities import regularise_Σ, print_err

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg

class SingleGaussian(Model):

    def __init__(self, data, verbose=None, independent_vars=True):
        Model.__init__(self, data, verbose=verbose)
        self.μ = ma.mean(self.X, axis=0).data

        if independent_vars:
            self.var_func = lambda x: np.diag(ma.var(x, axis=0).data)
        else:
            self.var_func = lambda x: ma.cov(x, rowvar=False).data

        self.Σ = self.var_func(self.X)
        if self.Σ.shape == ():
            self.Σ = np.array([[self.Σ]])

        # if there are no observations in any column of X then use mean 0.0, var 1
        self.μ[np.isnan(self.μ)] = 0
        self.Σ[np.isnan(self.Σ)] = 1

        # make sure the the cov matrix is pos def
        self.Σ = regularise_Σ(self.Σ)


        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        # fit the model to the data
        best_lls = self.lls

        if self.verbose: print_err("Fitting single gaussian using EM:")
        for i in range(max_iters):
            old_μ, old_Σ, old_expected_X = self.μ.copy(), self.Σ.copy(), self.expected_X.copy()

            # re-estimate the paramters μ and Σ (M-step)
            self.μ = np.mean(self.expected_X, axis=0)
            self.Σ = self.var_func(self.expected_X)
            if self.Σ.shape == ():
                self.Σ = np.array([[self.Σ]])
            self.Σ = regularise_Σ(self.Σ)
            
            # using the current parameters, estimate the values of the missing data (E-step)
            self._calc_ML_est()

            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if np.sum(self.lls[self.X.mask]) - np.sum(best_lls[self.X.mask]) < ϵ:
                self.μ, self.Σ, self.expected_X = old_μ, old_Σ, old_expected_X
                self.lls = best_lls
                break
            
            best_lls = self.lls

            if self.verbose: print_err("Iter: %s\t\t\tAvg LL: %f" % (i, np.mean(self.lls[self.X.mask])))
            
    def _calc_ML_est(self):
        expected_X = self.X.data.copy()

        expected_X[self.X.mask] = np.stack([self.μ]*self.N, axis=0)[self.X.mask]
        
        # take into account the conditional dependence
        for n in range(self.N):
            x_row = expected_X[n, :]
            mask_row = self.X.mask[n, :]

            # if there are no missing values or only missing then go to next iter
            if np.all(~mask_row) or np.all(mask_row): continue

            # calculate the mean of m|o
            # get the subsets of the covaraince matrice
            Σoo = self.Σ[np.ix_(~mask_row, ~mask_row)]
            Σmo = self.Σ[np.ix_(mask_row, ~mask_row)]
            if Σoo.shape != ():
                μmo = Σmo @ linalg.inv(Σoo) @ (x_row[~mask_row] - self.μ[~mask_row])

                # μmo will be 0 if the rows are indepenent
                expected_X[n, mask_row] += μmo

        self.expected_X = expected_X

    def _calc_ll(self):
        Λ = linalg.inv(self.Σ)

        for d in range(self.D):
            mask_row = np.array([False]*self.D)
            mask_row[d] = True     
            σ2 = linalg.inv(Λ[np.ix_(mask_row, mask_row)])
            Λtmp = σ2 @ Λ[np.ix_(mask_row, ~mask_row)] 
            
            for n in range(self.N):
                x_row = self.expected_X[n, :]
                μ = self.μ[mask_row] - Λtmp @ (x_row[~mask_row] - self.μ[~mask_row])

                # calculate ll
                self.lls[n, d] = np.log(stats.multivariate_normal.pdf(x_row[d], mean=μ, cov=σ2))

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data.copy()]*num_samples, axis=0)

        for n in range(self.N):
            # figure out the conditional distribution for the missing data given the observed data
            x_row = self.X[n, :]
            mask_row = self.X.mask[n, :]
            # if there are no missing values then go to next iter
            if np.all(~mask_row): continue            

            # the mean and var of the missing values
            μmo = self.μ[mask_row]
            Σmm = linalg.inv(linalg.inv(self.Σ)[np.ix_(mask_row, mask_row)])

            if np.any(~mask_row):
                # update the mean and covariance based on the conditional dependence between variables
                Σoo = self.Σ[np.ix_(~mask_row, ~mask_row)]
                Σmo = self.Σ[np.ix_(mask_row, ~mask_row)]
                diff = x_row[~mask_row] - self.μ[~mask_row]
                Σcond = Σmo @ linalg.inv(Σoo)

                μmo += Σcond @ diff # won't change if the variables are independent

            for i in range(num_samples):
                sampled_Xs[i, n, mask_row] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1, random_state=i)

        return sampled_Xs
