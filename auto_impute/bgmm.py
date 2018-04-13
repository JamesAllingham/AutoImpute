# James Allingham
# April 2018
# bgmm.py
# Imputation using a Gaussian Mixture Model fitted using variational Bayes

from model import Model
from utilities import regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from scipy import special
from sklearn.cluster import KMeans

class BGMM(Model):

    def __init__(self, data, num_gaussians, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians

        # hyper-parameters
        self.α0 = 1e-3
        self.m0 = np.zeros(shape=(self.num_features, ))
        self.β0 = 1
        self.W0 = np.eye(self.num_features)
        self.ν0 = self.num_features

        self.rs = np.zeros(shape=(self.N, self.num_gaussians))

        # intial params
        # use k-means to initialise the means and covs
        self.ms  = [np.zeros(shape=(self.num_features,))]*self.num_gaussians
        self.Ws = [np.zeros(shape=(self.num_features, self.num_features))]*self.num_gaussians
        mean_imputed_X = self.X.data.copy()
        mean_imputed_X[self.X.mask] = ma.mean(self.X, axis=0)[np.where(self.X.mask)[1]]
        # mean_imputed_X += np.random.random(mean_imputed_X.shape)*0.01
        kmeans = KMeans(n_clusters=self.num_gaussians, random_state=0).fit(mean_imputed_X)
        self.rs[np.arange(self.N), kmeans.labels_] = 1
        # for k in range(num_gaussians):
        #     locs = np.where(kmeans.labels_ == k)[0]
        #     self.ms[k] = np.mean(mean_imputed_X[locs, :], axis=0)
        #     self.Ws[k] = np.cov(mean_imputed_X[locs, :], rowvar=False)
        
        # Ns = np.sum(self.rs, axis=0)

        # self.αs = [Ns[k] + self.α0 for k in range(self.num_gaussians)]

        self.αs = [self.α0]*self.num_gaussians
        self.βs = [self.β0]*self.num_gaussians
        self.νs = [self.ν0]*self.num_gaussians

        self._calc_updated_params()

    def fit(self, max_iters=100, ϵ=1e-1):
        if self.verbose: print("Fitting model:")

        for i in range(max_iters):
            old_αs, old_ms, old_βs, old_Ws, old_νs = self.αs, self.ms, self.βs, self.Ws, self.νs

            # do one iteration of inference
            self._calc_rs()
            self._calc_updated_params()

            # update the ML imputation and the LL
            self._calc_ML_est()
            self._calc_ll()
            
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

            if np.linalg.norm(old_αs - np.array(self.αs)) + np.linalg.norm(old_ms - np.array(self.ms)) + np.linalg.norm(old_βs - np.array(self.βs)) \
                + np.linalg.norm(old_Ws - np.array(self.Ws)) + np.linalg.norm(old_νs - np.array(self.νs)) <= ϵ:
                break

    # "E-step"
    def _calc_rs(self):
        log_πs = special.digamma(self.αs) - special.digamma(np.sum(self.αs))
        log_ps = np.stack([log_πs]*self.N, axis=0)
        for n in range(self.N):
            o_locs = np.where(~self.X[n, :].mask)[0]
            
            if not o_locs.size: continue

            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            x_row = self.X[n,:].data
            log_ps[n, :] += 0.5*np.log(np.array([linalg.det(self.Ws[k][oo_coords].reshape(o_locs.size, o_locs.size)) for k in range(self.num_gaussians)]))

            for k in range(self.num_gaussians):
                log_ps[n, k] += 0.5*np.sum([special.digamma(0.5*(self.νs[k] - self.num_features + o_locs.size + 1 - i)) for i in range(o_locs.size)]) 

            for k in range(self.num_gaussians):
                tmp = 0.5*self.βs[k]
                tmp *= (self.νs[k] - self.num_features + o_locs.size)
                tmp *= (x_row[o_locs] - self.ms[k][o_locs]).T
                tmp = tmp @ self.Ws[k][oo_coords].reshape(o_locs.size, o_locs.size)
                tmp = tmp @ (x_row[o_locs] - self.ms[k][o_locs])
                log_ps[n, k] -= tmp

        ps = np.exp(log_ps) + 1e-9 # incase none of the components want to take charge of an example
        self.rs = ps/np.sum(ps, axis=1, keepdims=True)        

    # "M-step"
    def _calc_updated_params(self):
        Ns = np.sum(self.rs, axis=0)

        self.αs = [Ns[k] + self.α0 for k in range(self.num_gaussians)]

        prev_ms = self.ms.copy()
        
        for k in range(self.num_gaussians):
            # check if the component is still valid
            if Ns[k] == 0: continue

            # build a repaired version of the data to use for the subsequent calculations
            x_rep = self.X.data.copy()
            for n in range(self.N):
                m_locs = np.where(self.X[n, :].mask)
                x_rep[n, m_locs] = prev_ms[k][m_locs]

            x_bar = 1/Ns[k]*np.sum(self.rs[:, k, np.newaxis] * x_rep, axis=0)

            self.βs[k] = self.β0 + Ns[k]
            self.νs[k] = self.ν0 + Ns[k]
            self.ms[k] = 1/self.βs[k]*(self.β0*self.m0 + Ns[k]*x_bar)
            W_inv = linalg.inv(self.W0)
            W_inv += 1/Ns[k]*np.einsum('ij,ikl->kl', self.rs[:, k, np.newaxis], np.einsum('ij,ik->ijk' ,x_rep - x_bar, x_rep - x_bar))
            W_inv += self.β0*Ns[k]/(self.β0 + Ns[k])*np.outer(x_bar - self.m0, x_bar - self.m0)
            self.Ws[k] = np.linalg.inv(W_inv)

    def _calc_ML_est(self):
        # Note: the expectation for each mean is simply self.ms[k],
        # similarly, the expectation for the precision is self.νs[k]*self.Ws[k]
        Xs = np.stack([self.X.data]*self.num_gaussians, axis=0)

        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs = np.where(~mask_row)[0]
            m_locs = np.where(mask_row)[0]
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))

            for k in range(self.num_gaussians):
                μ_k = self.ms[k]
                Λ_k = self.νs[k]*self.Ws[k]
                Xs[k, n, m_locs] = μ_k[m_locs]

                if o_locs.size:
                    Σ = np.linalg.inv(Λ_k[mm_coords].reshape(m_locs.size, m_locs.size))
                    Xs[k, n, m_locs] -= Σ @ Λ_k[mo_coords].reshape(m_locs.size, o_locs.size) @ (x_row[o_locs] - μ_k[o_locs])
                
        self.expected_X = np.zeros_like(self.X.data)
        for k in range(self.num_gaussians):
            for n in range(self.N):
                self.expected_X[n, :] += self.rs[n, k]*Xs[k, n, :]
    
    def _calc_ll(self):
        ll = 0
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs = np.where(~mask_row)[0]
            m_locs = np.where(mask_row)[0]
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))

            tmp = 0
            for k in range(self.num_gaussians):
                μ_k = self.ms[k]
                Λ_k = self.νs[k]*self.Ws[k]

                # now the mean and var for the missing data given the seen data
                var = np.linalg.inv(Λ_k[mm_coords].reshape(m_locs.size, m_locs.size))
                mean = μ_k[m_locs]
                if o_locs.size:
                    mean -= var @ Λ_k[mo_coords].reshape(m_locs.size, o_locs.size) @ (x_row[o_locs] - μ_k[o_locs])

                tmp += self.rs[n, k] * stats.multivariate_normal.pdf(self.expected_X[n, m_locs], mean=mean, cov=var)

            ll += np.log(tmp)
        self.ll = ll/self.N

    def sample(self, n):
        pass
