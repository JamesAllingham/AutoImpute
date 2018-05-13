# James Allingham
# March 2018
# gmm.py
# Imputation using a Gaussian Mixture Model fitted using the EM algorithm
# Based on Delalleau, O., Courville, A., & Bengio, Y. (n.d.). Efficient EM Training of Gaussian Mixtures with Missing Data. 
# Retrieved from https://arxiv.org/pdf/1209.0521.pdf
# and Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

from model import Model
from utilities import get_locs_and_coords, regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from sklearn.cluster import KMeans

class GMM(Model):

    def __init__(self, data, num_components, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_components = num_components

        # use k-means to initialise params
        self.rs = np.zeros(shape=(self.N, self.num_components))
        mean_imputed_X = self.X.data.copy()
        mean_imputed_X[self.X.mask] = ma.mean(self.X, axis=0)[np.where(self.X.mask)[1]]
        kmeans = KMeans(n_clusters=self.num_components, random_state=0).fit(mean_imputed_X)
        self.rs[np.arange(self.N), kmeans.labels_] = 1
        self.μs = np.stack([np.mean(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0) for k in range(self.num_components)], axis=0)
        # self.Σs = np.stack([np.cov(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], rowvar=False) for k in range(self.num_components)], axis=0)
        self.Σs = np.stack([regularise_Σ(np.diag(np.var(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0))) for k in range(self.num_components)], axis=0)

        self.rs = np.random.rand(self.N, self.num_components)
        self.rs = self.rs/np.sum(self.rs, axis=1, keepdims=True)
        # self.rs = self.rs + np.random.rand(*self.rs.shape)*1e-1
        # print(self.rs)

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μs, old_Σs, old_expected_X, old_rs = self.μs.copy(), self.Σs.copy(), self.expected_X.copy(), self.rs.copy()

            # E-step
            self._calc_rs()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.μs, self.Σs, self.expected_X, self.rs = old_μs, old_Σs, old_expected_X, old_rs
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
            
            o_locs, _, oo_coords, _, _, _ = get_locs_and_coords(mask_row)

            x = x_row[o_locs]
            sz = len(x)
            if sz:
                for k in range(self.num_components):
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(sz, sz)
                    μo = self.μs[k, o_locs]

                    rs[n, k] = stats.multivariate_normal.pdf(x, mean=μo, cov=Σoo, allow_singular=True)
                    
            else: # not actually too sure how to handle this situation
                rs[n, :] = np.mean(self.rs, axis=0)

        self.rs = rs/np.sum(rs, axis=1, keepdims=True)

    # M-step
    def _update_params(self):
        # first fill in the missing values with each gaussian
        self._calc_ML_est()  
        
        for k in range(self.num_components):
            # first fill in the missing elements of X
            X = self.X.data
            μ = np.stack([self.μs[k]]*self.N, axis=0)
            X[self.X.mask] = μ[self.X.mask]             
            for n in range(self.N):
                x_row = self.X[n, :].data
                mask_row = self.X[n, :].mask

                if np.all(~mask_row): continue

                o_locs, m_locs, oo_coords, _, mo_coords, _ = get_locs_and_coords(mask_row)

                if o_locs.size:
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[k, o_locs]
                    X[n, m_locs] += Σmo @ linalg.inv(Σoo) @ diff

            # now recompute μs
            p = self.rs[:, k]
            self.μs[k] = (p @ X)/np.sum(p)

            # and now Σs
            # calc C
            C = np.zeros(shape=(self.num_features, self.num_features))
            for n in range(self.N):
                mask_row = self.X[n, :].mask

                if np.all(~mask_row): continue

                o_locs, m_locs, oo_coords, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

                Σmm = self.Σs[k, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                tmp = Σmm
                if o_locs.size:
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs)) 
                    tmp -= Σmo @ linalg.inv(Σoo) @ Σmo.T

                tmp = p[n]/np.sum(p)*tmp
                C[mm_coords] += tmp.reshape(len(m_locs)**2)
                

            self.Σs[k] = np.zeros_like(C)
            for n in range(self.N):
                x_row = self.X[n, :].data
                diff = x_row - self.μs[k]
                # self.Σs[k] += np.outer(diff, diff.T)*p[n]
                self.Σs[k] += np.diag(diff * diff)*p[n]

            # print(self.Σs[k])
            self.Σs[k] /= np.sum(p)
            # print(self.Σs[k])
            self.Σs[k] += C
            # regularisation term ensuring that the cov matrix is always pos def
            self.Σs[k] = regularise_Σ(self.Σs[k])
            # print(self.Σs[k])
            # print(C)

    def _calc_ML_est(self):
        self.expected_X = self.X.data.copy()

        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs, m_locs, oo_coords, _, mo_coords, _ = get_locs_and_coords(mask_row)

            k = np.argmax(self.rs[n, :])

            self.expected_X[n, m_locs] = self.μs[k, m_locs]
            
            if o_locs.size:
                Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                
                diff = x_row[o_locs] - self.μs[k, o_locs]
                self.expected_X[n, m_locs] += Σmo @ linalg.inv(Σoo) @ diff

    def _calc_ll(self):
        lls = []
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs, m_locs, oo_coords, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

            prob = 0
            for k in range(self.num_components):
                # now the mean and var for the missing data given the seen data
                μmo = self.μs[k, m_locs]
                
                if o_locs.size:
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[k,o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σs[k, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                prob += self.rs[n, k] * stats.multivariate_normal.pdf(self.expected_X[n, m_locs], mean=μmo, cov=Σmm, allow_singular=True)

            lls.append(np.log(prob))
        self.ll = np.mean(lls)

    def sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for n in range(self.N):
            # figure out the conditional distribution for the missing data given the observed data
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask
            # if there are no missing values then go to next iter
            if np.all(~mask_row): continue

            # figure out which values are missing
            o_locs, m_locs, oo_coords, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

            for i in range(num_samples):
                choice = np.random.choice(self.num_components, p=self.rs[n, :])
                μmo = self.μs[choice, m_locs]

                if o_locs.size:
                    Σoo = self.Σs[choice, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[choice, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[choice,o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σs[choice, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                sampled_Xs[i, n, m_locs] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

        return sampled_Xs
        