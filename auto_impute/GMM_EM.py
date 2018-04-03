# James Allingham
# March 2018
# GMM_EM.py
# Imputation using a Gaussian Mixture Model fitted using the EM algorithm

from Model import Model

import numpy as np
from scipy import stats
from scipy import linalg

class GMM(Model):

    def __init__(self, data, num_gaussians, ϵ=None, max_iters=None, verbose=None):
        Model.__init__(self, data, ϵ=ϵ, max_iters=max_iters, verbose=verbose)        
        self.num_gaussians = num_gaussians
        # self.μs = np.random.rand(self.num_gaussians, self.num_features)
        # self.Σs = np.stack([np.eye(self.num_features) for _ in range(self.num_gaussians)], axis=0)
        indices = np.stack([np.random.choice(self.N, int(self.N/2)) for _ in range(self.num_gaussians)], axis=0)
        self.μs = np.stack([np.nanmean(self.X[idx, :], axis=0) for idx in indices], axis=0)
        # self.Σs = np.stack([np.diag(self.μs[j,:]) for j in range(self.num_gaussians)], axis=0)
        self.Σs = np.stack(
            [np.nanmean(
                [np.outer(self.X[i, :] - μ, self.X[i, :] - μ) for i in idx], axis=0) for μ, idx in zip(self.μs, indices)]
            , axis=0)

        self.Xs = np.array([])
        self.ps = np.array([])

        best_ll = -np.inf

        if (self.verbose): print("Fitting model:")
        for k in range(self.max_iters):
            old_μs, old_Σs, old_expected_X, old_Xs, old_ps = self.μs.copy(), self.Σs.copy(), self.expected_X.copy(), self.Xs.copy(), self.ps.copy()

            # E-step
            qs = np.zeros(shape=(self.N, self.num_gaussians))
            for i in range(self.N):
                x_row = self.X[i, :]
                o_locs = np.where(~np.isnan(x_row))[0]
                oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))

                x = x_row[o_locs]
                sz = len(x)
                if (sz):
                    for j in range(self.num_gaussians):
                        Σoo = self.Σs[j, :, :][oo_coords].reshape(sz, sz)
                        μo = self.μs[j, o_locs]

                        qs[i, j] = stats.multivariate_normal.pdf(x, mean=μo, cov=Σoo)
                        
                else: # not actually too sure how to handle this situation
                    qs[i, :] = np.random.random(num_gaussians)

            self.ps = qs/np.sum(qs, axis=1, keepdims=True)
            # print(self.ps)
            # print(qs)

            # M-step
            # first fill in the missing values with each gaussian
            self.__calc_expectation()  
            
            # now recompute μs
            for j in range(self.num_gaussians):
                p = self.ps[:, j]
                self.μs[j] = (p @ self.Xs[j])/np.sum(p)

            # and now Σs
            for j in range(self.num_gaussians):

                p = self.ps[:, j]

                # calc C
                C = np.zeros(shape=(self.num_features, self.num_features))
                for i in range(self.N):
                    x_row = self.X[i, :]

                    if np.all(~np.isnan(x_row)): continue

                    o_locs = np.where(~np.isnan(x_row))[0]
                    m_locs = np.where(np.isnan(x_row))[0]
                    oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
                    mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
                    mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

                    Σmm = self.Σs[j, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                    tmp = Σmm
                    if (o_locs.size):
                        Σoo = self.Σs[j, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                        Σmo = self.Σs[j, :, :][mo_coords].reshape(len(m_locs), len(o_locs)) 
                        tmp -= Σmo @ linalg.inv(Σoo) @ Σmo.T

                    tmp = p[i]/np.sum(p)*tmp
                    C[mm_coords] += tmp.reshape(len(m_locs)**2)
                    

                self.Σs[j] = np.zeros_like(C)
                for i in range(self.N):
                    diff = self.Xs[j, i, :] - self.μs[j]
                    self.Σs[j] += np.outer(diff, diff.T)*p[i]

                self.Σs[j] /= np.sum(p)
                self.Σs[j] += C
                # regularisation term ensuring that the cov matrix is always pos def
                self.Σs[j] += np.eye(self.num_features)*1
            
            self.__calc_expectation()
            # if the log likelihood stops improving then stop iterating
            self.__log_likelihood()
            if (self.ll < best_ll or self.ll - best_ll < self.ϵ):
                self.μs, self.Σs, self.expected_X, self.Xs, self.ps = old_μs, old_Σs, old_expected_X, old_Xs, old_ps
                self.ll = best_ll
                break
            
            best_ll = self.ll
            if (self.verbose): print("Iter: %s\t\tLL: %f" % (k, self.ll))

    def __calc_expectation(self): # should probably split this into two functions one for expected_X and one for Xs
        Xs = np.stack([self.X]*self.num_gaussians, axis=0)

        for i in range(self.N):
            x_row = self.X[i, :]

            if np.all(~np.isnan(x_row)): continue

            o_locs = np.where(~np.isnan(x_row))[0]
            m_locs = np.where(np.isnan(x_row))[0]
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))

            for j in range(self.num_gaussians):
                diff = x_row[o_locs] - self.μs[j, o_locs]

                Xs[j, i, m_locs] = self.μs[j, m_locs]
                if (len(o_locs)):
                    Σoo = self.Σs[j, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[j, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    Xs[j, i, m_locs] += Σmo @ linalg.inv(Σoo) @ diff

                
        self.expected_X = np.zeros_like(self.X)
        for j in range(self.num_gaussians):
            for i in range(self.N):
                self.expected_X[i, :] += self.ps[i, j]*Xs[j, i, :]
            
        self.Xs = Xs

    def __log_likelihood(self):
        ll = 0
        for i in range(self.N):
            tmp = 0
            for j in range(self.num_gaussians):
                tmp += self.ps[i, j] * stats.multivariate_normal.pdf(self.expected_X[i, :], mean=self.μs[j, :], cov=self.Σs[j, :, :])
            ll += np.log(tmp)
        self.ll = ll/self.N

    def sample(self, n):
        sampled_Xs = np.stack([self.X]*n, axis=0)

        for i in range(self.N):
            # figure out the conditional distribution for the missing data given the observed data
            x_row = self.X[i,:]
            # if there are no missing values then go to next iter
            if np.all(~np.isnan(x_row)): continue

            # figure out which values are missing
            o_locs = np.where(~np.isnan(x_row))[0]
            m_locs = np.where(np.isnan(x_row))[0]
            mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))

            for j in range(n):
                choice = np.random.choice(self.num_gaussians, p=self.ps[i, :])
                μmo = self.μs[choice,m_locs]

                if (len(o_locs)):
                    Σoo = self.Σs[choice, :, :][oo_coords].reshape(len(o_locs), len(o_locs))
                    Σmo = self.Σs[choice, :, :][mo_coords].reshape(len(m_locs), len(o_locs))
                    diff = x_row[o_locs] - self.μs[choice,o_locs]
                    μmo += Σmo @ linalg.inv(Σoo) @ diff

                Σmm = self.Σs[choice, :, :][mm_coords].reshape(len(m_locs), len(m_locs))

                sampled_Xs[j, i, m_locs] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

        # return sampled_Xs*self.std + self.mean
        return sampled_Xs
        