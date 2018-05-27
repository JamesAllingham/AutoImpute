# James Allingham
# March 2018
# gmm.py
# Imputation using a Gaussian Mixture Model fitted using the EM algorithm
# Based on Delalleau, O., Courville, A., & Bengio, Y. (n.d.). Efficient EM Training of Gaussian Mixtures with Missing Data. 
# Retrieved from https://arxiv.org/pdf/1209.0521.pdf
# and Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

from model import Model
from mi import MeanImpute
from utilities import regularise_Σ, print_err

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from sklearn.cluster import KMeans

class GMM(Model):

    def __init__(self, data, num_components, verbose=None, independent_vars=True, α0=None, m0=None, β0=None, W0=None, ν0=None, map_est=True):
        Model.__init__(self, data, verbose=verbose)
        self.num_components = num_components
        self.independent_vars = independent_vars

        self.map_est = map_est

        if independent_vars:
            self.var_func = lambda x: np.diag(ma.var(x, axis=0))
        else:
            self.var_func = lambda x: ma.cov(x, rowvar=False).data

        def min_2d(x):
            if x.shape == ():
               return np.array([[x]])
            else:
                return x

        # hyper-parameters
        if α0 is not None:
            self.α0 = α0
        else:
            self.α0 = 1

        if m0 is not None:
            self.m0 = m0
        else:
            self.m0 = np.zeros(shape=(self.D, ))
        
        if β0 is not None:
            self.β0 = β0
        else:
            self.β0 = 1e-0

        if W0 is not None:
            self.W0 = W0
        else:
            self.W0 = np.eye(self.D)
        
        if ν0 is not None:
            self.ν0 = ν0
        else:
            self.ν0 = self.D

        # use k-means to initialise params
        rs = np.zeros(shape=(self.N, self.num_components))
        mi_model = MeanImpute(self.X, verbose=None)
        mean_imputed_X = mi_model.ml_imputation()
        
        kmeans = KMeans(n_clusters=self.num_components, random_state=0).fit(mean_imputed_X)
        rs[np.arange(self.N), kmeans.labels_] = 1
        
        self.πs = np.mean(rs, axis=0)
        self.μs = np.stack([np.mean(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0) for k in range(self.num_components)], axis=0)
        self.Σs = np.stack([min_2d(self.var_func(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :])) for k in range(self.num_components)], axis=0)
        for k in range(self.num_components):
            # handle edge cases wherer no values are assigned to a component
            self.Σs[k][np.isnan(self.Σs[k])] =  0
            self.μs[k][np.isnan(self.μs[k])] = 0
            # handle edge cases where only 1 value us assigned to a component
            self.Σs[k] = regularise_Σ(self.Σs[k])

        self.rs = np.random.rand(self.N, self.num_components)
        self.rs = self.rs/np.sum(self.rs, axis=1, keepdims=True)

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_lls = self.lls
        if self.verbose: print_err("Fitting GMM using EM algorithm (%s):" % ("MLE" if not self.map_est else "MAP estimate",))
        for i in range(max_iters):
            old_μs, old_Σs, old_πs, old_expected_X, old_rs = self.μs.copy(), self.Σs.copy(), self.πs.copy(), self.expected_X.copy(), self.rs.copy()

            # E-step
            self._calc_rs()
            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if np.sum(self.lls[self.X.mask]) - np.sum(best_lls[self.X.mask]) < ϵ:
                self.μs, self.Σs, self.πs, self.expected_X, self.rs = old_μs, old_Σs, old_πs, old_expected_X, old_rs
                self.lls = best_lls
                break
            
            best_lls = self.lls
            if self.verbose: print_err("Iter: %s\t\t\tAvg LL: %f" % (i, np.mean(self.lls[self.X.mask])))

    # E-step
    def _calc_rs(self):
        rs = np.zeros(shape=(self.N, self.num_components))
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            x = x_row[~mask_row]
            sz = len(x)
            if sz:
                for k in range(self.num_components):
                    Σoo = self.Σs[k][np.ix_(~mask_row, ~mask_row)]
                    μo = self.μs[k, ~mask_row]

                    rs[n, k] = self.πs[k]*stats.multivariate_normal.pdf(x, mean=μo, cov=Σoo, allow_singular=True)
                    
            else: # not actually too sure how to handle this situation
                rs[n, :] = self.πs

        rs += 1e-64 # in case none of the components want to take charge of an example
        self.rs = rs/np.sum(rs, axis=1, keepdims=True)

    # M-step
    def _update_params(self): 

        # recompute πs
        self.πs = np.mean(self.rs, axis=0)
        αs = np.array([self.α0]*self.num_components)

        # now the other parameters that depend on X
        for k in range(self.num_components):
            # first fill in the missing elements of X
            X = self.X.data
            μ = np.stack([self.μs[k]]*self.N, axis=0)
            X[self.X.mask] = μ[self.X.mask]             
            
            # take conditional dependence into account
            for n in range(self.N):
                x_row = self.X[n, :].data
                mask_row = self.X[n, :].mask

                # if there are no missing values or only missing then go to next iter
                if np.all(~mask_row) or np.all(mask_row): continue
                
                Σoo = self.Σs[k][np.ix_(~mask_row, ~mask_row)]
                Σmo = self.Σs[k][np.ix_(mask_row, ~mask_row)]
                diff = x_row[~mask_row] - self.μs[k, ~mask_row]
                X[n, mask_row] += Σmo @ linalg.inv(Σoo) @ diff

            # now recompute μs
            p = self.rs[:, k]
            self.μs[k] = (p @ X)/np.sum(p)
            self.μs[k][np.isnan(self.μs[k])] = 0

            # and now Σs
            # calc C
            C = np.zeros(shape=(self.D, self.D))
            for n in range(self.N):
                mask_row = self.X[n, :].mask

                if np.all(~mask_row): continue

                Σmm = self.Σs[k][np.ix_(mask_row, mask_row)]

                tmp = Σmm
                if np.any(~mask_row):
                    Σoo = self.Σs[k][np.ix_(~mask_row, ~mask_row)]
                    Σmo = self.Σs[k][np.ix_(mask_row, ~mask_row)] 
                    tmp -= Σmo @ linalg.inv(Σoo) @ Σmo.T

                tmp = p[n]/np.sum(p)*tmp
                C[np.ix_(mask_row, mask_row)] += tmp
                

            self.Σs[k] = np.zeros_like(C)
            for n in range(self.N):
                x_row = self.X[n, :].data
                diff = x_row - self.μs[k]
                if self.independent_vars:
                    self.Σs[k] += np.diag(diff * diff)*p[n]
                else:
                    self.Σs[k] += np.outer(diff, diff.T)*p[n]

            self.Σs[k] /= np.sum(p)
            self.Σs[k] += C
            self.Σs[k][np.isnan(self.Σs[k])] =  0
            # regularisation term ensuring that the cov matrix is always pos def
            self.Σs[k] = regularise_Σ(self.Σs[k])

            # now if we want a MAP estimate rather than the MLE, we can use these statistics calcualted above to update prior beliefs
            if self.map_est:
                # we need one more statistic N_k
                N_k = np.sum(self.rs[:, k])

                # update the priors
                αs[k] = self.α0 + N_k
                β = self.β0 + N_k
                m = (self.β0*self.m0 + N_k*self.μs[k])/(self.β0 + N_k)
                ν = self.ν0 + N_k
                W = self.W0 + self.Σs[k] + self.β0*N_k/(self.β0 + N_k)*np.diag((self.μs[k] - self.m0)**2)

                # now since we are doing a MAP estimate we take the mode of the posterior distributions to get out estiamtes
                self.μs[k] = m
                self.Σs[k] = W/(ν + self.D + 1)
        
        if self.map_est:
            self.πs = (αs - 1)/np.sum(αs - 1)

    def _calc_ML_est(self):
        self.expected_X = self.X.data.copy()

        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue            

            k = np.argmax(self.rs[n, :])

            self.expected_X[n, mask_row] = self.μs[k, mask_row]

            if np.any(~mask_row):
                Σoo = self.Σs[k][np.ix_(~mask_row, ~mask_row)]
                Σmo = self.Σs[k][np.ix_(mask_row, ~mask_row)]
                
                diff = x_row[~mask_row] - self.μs[k, ~mask_row]
                self.expected_X[n, mask_row] += Σmo @ linalg.inv(Σoo) @ diff

    def _calc_ll(self):
        self.lls = np.zeros_like(self.lls)
        for k in range(self.num_components):
            Λ = linalg.inv(self.Σs[k])

            for d in range(self.D):
                mask_row = np.array([False]*self.D)
                mask_row[d] = True     
                σ2 = linalg.inv(Λ[np.ix_(mask_row, mask_row)])
                Λtmp = σ2 @ Λ[np.ix_(mask_row, ~mask_row)] 
                
                for n in range(self.N):
                    x_row = self.expected_X[n, :]
                    μ = self.μs[k][mask_row] - Λtmp @ (x_row[~mask_row] - self.μs[k][~mask_row])

                    # calculate ll
                    self.lls[n, d] += self.πs[k]*stats.multivariate_normal.pdf(x_row[d], mean=μ, cov=σ2)

        self.lls = np.log(self.lls)

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):
            k = np.random.choice(self.num_components, p=self.πs)            
        
            for n in range(self.N):
                x_row = self.X[n, :].data
                mask_row = self.X[n, :].mask


                # if there are no missing values then go to next iter
                if np.all(~mask_row): continue

                μmo = self.μs[k, mask_row]
                Σmm = linalg.inv(linalg.inv(self.Σs[k])[np.ix_(mask_row, mask_row)])

                if np.any(~mask_row):
                    Σoo = self.Σs[k][np.ix_(~mask_row, ~mask_row)]
                    Σmo = self.Σs[k][np.ix_(mask_row, ~mask_row)]
                    diff = x_row[~mask_row] - self.μs[k, ~mask_row]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                sampled_Xs[i, n, mask_row] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

        return sampled_Xs
        