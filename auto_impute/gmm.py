# James Allingham
# March 2018
# gmm.py
# Imputation using a Gaussian Mixture Model fitted using the EM algorithm

from model import Model
from utilities import regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg

class GMM(Model):

    def __init__(self, data, num_gaussians, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians
        indices = np.stack([np.random.choice(self.N, int(self.N/2)) for _ in range(self.num_gaussians)], axis=0)
        self.μs = np.stack([ma.mean(self.X[idx, :], axis=0).data for idx in indices], axis=0)
        self.Σs = np.stack([regularise_Σ(ma.cov(self.X[idx, :], rowvar=False).data) for idx in indices], axis=0)

        self.Xs = np.array([])
        self.ps = np.random.rand(self.N, self.num_gaussians)

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μs, old_Σs, old_expected_X, old_Xs, old_ps = self.μs.copy(), self.Σs.copy(), self.expected_X.copy(), self.Xs.copy(), self.ps.copy()

            # E-step
            qs = np.zeros(shape=(self.N, self.num_gaussians))
            for n in range(self.N):
                x_row = self.X[n, :].data
                mask_row = self.X[n, :].mask
                o_locs = np.where(~mask_row)[0]
                oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))

                x = x_row[o_locs]
                sz = len(x)
                if sz:
                    for k in range(self.num_gaussians):
                        Σoo = self.Σs[k, :, :][oo_coords].reshape(sz, sz)
                        μo = self.μs[k, o_locs]

                        qs[n, k] = stats.multivariate_normal.pdf(x, mean=μo, cov=Σoo)
                        
                else: # not actually too sure how to handle this situation
                    qs[n, :] = np.mean(self.ps, axis=0)

            self.ps = qs/np.sum(qs, axis=1, keepdims=True)

            # M-step
            # first fill in the missing values with each gaussian
            self._calc_ML_est()  
            
            # now recompute μs
            for k in range(self.num_gaussians):
                p = self.ps[:, k]
                self.μs[k] = (p @ self.Xs[k])/np.sum(p)

            # and now Σs
            for k in range(self.num_gaussians):

                p = self.ps[:, k]

                # calc C
                C = np.zeros(shape=(self.num_features, self.num_features))
                for n in range(self.N):
                    x_row = self.X[n, :].data
                    mask_row = self.X[n, :].mask

                    if np.all(~mask_row): continue

                    o_locs = np.where(~mask_row)[0]
                    m_locs = np.where(mask_row)[0]
                    oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
                    mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
                    mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

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
                    diff = self.Xs[k, n, :] - self.μs[k]
                    self.Σs[k] += np.outer(diff, diff.T)*p[n]

                self.Σs[k] /= np.sum(p)
                self.Σs[k] += C
                # regularisation term ensuring that the cov matrix is always pos def
                self.Σs[k] += np.eye(self.num_features)*1e-3
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.μs, self.Σs, self.expected_X, self.Xs, self.ps = old_μs, old_Σs, old_expected_X, old_Xs, old_ps
                self.ll = best_ll
                break
            
            best_ll = self.ll
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

    def _calc_ML_est(self): # should probably split this into two functions one for expected_X and one for Xs
        Xs = np.stack([self.X.data]*self.num_gaussians, axis=0)

        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs = np.where(~mask_row)[0]
            m_locs = np.where(mask_row)[0]
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))

            for k in range(self.num_gaussians):
                diff = x_row[o_locs] - self.μs[k, o_locs]

                Xs[k, n, m_locs] = self.μs[k, m_locs]
                if o_locs.size:
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    Xs[k, n, m_locs] += Σmo @ linalg.inv(Σoo) @ diff

                
        self.expected_X = np.zeros_like(self.X.data)
        for k in range(self.num_gaussians):
            for n in range(self.N):
                self.expected_X[n, :] += self.ps[n, k]*Xs[k, n, :]
            
        self.Xs = Xs

    def _calc_ll(self):
        lls = []
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs = np.where(~mask_row)[0]
            m_locs = np.where(mask_row)[0]
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

            tmp = 0
            for k in range(self.num_gaussians):
                # now the mean and var for the missing data given the seen data
                μmo = self.μs[k, m_locs]
                
                if o_locs.size:
                    Σoo = self.Σs[k, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[k, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[k,o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σs[k, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                tmp += self.ps[n, k] * stats.multivariate_normal.pdf(self.expected_X[n, m_locs], mean=μmo, cov=Σmm)
            lls.append(np.log(tmp))
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
            o_locs = np.where(~mask_row)[0]
            m_locs = np.where(mask_row)[0]
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

            for i in range(num_samples):
                choice = np.random.choice(self.num_gaussians, p=self.ps[n, :])
                μmo = self.μs[choice, m_locs]

                if o_locs.size:
                    Σoo = self.Σs[choice, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[choice, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[choice,o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σs[choice, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                sampled_Xs[i, n, m_locs] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

        return sampled_Xs
        