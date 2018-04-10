# James Allingham
# April 2018
# bgmm.py
# Imputation using a Gaussian Mixture Model fitted using variational Bayes

from model import Model

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from scipy import special

class BGMM(Model):

    def __init__(self, data, num_gaussians, verbose=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians

        # hyper-parameters
        self.α0 = 1e-3
        self.m0 = np.zeros(shape=(self.num_features,))
        self.β0 = 1
        self.W0 = np.eye(self.num_features)
        self.ν0 = self.num_features

        self.rs = None

        # updated params
        self.αs = [self.α0]*self.num_gaussians
        self.ms = [self.m0]*self.num_gaussians
        self.βs = [self.β0]*self.num_gaussians
        self.Ws = [self.W0]*self.num_gaussians
        self.νs = [self.ν0]*self.num_gaussians

        self._calc_expectation()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        pass

    def _calc_rs(self):
        log_πs = special.digamma(self.αs) - special.digamma(np.sum(self.αs))
        
        log_ps = np.stack([log_πs]*self.N, axis=0)
        for n in range(self.N):
            o_locs = np.where(~self.X[n, :].mask)
            oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
            x_row = self.X[n,:].data
            log_ps[n, :] += 0.5*np.log(np.array([self.Ws[k][o_locs, o_locs] for k in self.num_gaussians]))
            log_ps[n, :] += np.array(
                    [0.5*np.sum(
                            [special.digamma(0.5*(self.νs[k] - self.num_features + len(o_locs) + 1 - i)) for i in range(len(o_locs))]
                        ) for k in self.num_gaussians
                    ]
                )
            log_ps[n, :] -= 0.5*np.array([self.βs[k]*(self.νs[k] - self.num_features + len(o_locs))*(x_row[o_locs] - self.ms[k][o_locs]).T @ self.Ws[k][oo_coords] @ (x_row[o_locs] - self.ms[k][o_locs]) for k in self.num_gaussians]) 

        ps = np.exp(log_ps)
        self.rs = ps/np.sum(ps, axis=1, keepdims=True)

    def _calc_updated_params(self):
        Ns = np.sum(self.rs, axis=0)
        prev_ms = self.ms.copy()
        
        for k in range(self.num_gaussians):
            # build a repaired version of the data to use for the subsequent calculations
            x_rep = self.X.data.copy()
            for n in range(self.N):
                m_locs = np.where(self.X[n, :].mask)
                x_rep[n, m_locs] = self.ms[k][m_locs]

            x_bar = 1/Ns[k]*np.sum(self.rs[k, :] * x_rep, axis=0)

            self.βs[k] = self.β0 + Ns[k]
            self.νs[k] = self.ν0 + Ns[k]
            self.ms[k] = 1/self.βs[k](self.β0*self.m0 + Ns[k]*x_bar)
            W_inv = linalg.inv(self.W0)
            W_inv += 1/Ns[k]*np.sum(self.rs*np.outer(x_rep - x_bar, x_rep - x_bar), axis=0)
            W_inv += self.β0*Ns[k]/(self.β0 + Ns[k])*np.outer(x_bar - self.m0, x_bar - self.m0)
            self.Ws[k] = np.linalg.inv(W_inv)

    def _calc_expectation(self):
        pass
    
    def _calc_ll(self):
        pass

    def sample(self, n):
        pass
