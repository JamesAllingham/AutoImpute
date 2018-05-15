# James Allingham
# April 2018
# bgmm.py
# Imputation using a Gaussian Mixture Model fitted using variational Bayes

from model import Model
from utilities import get_locs_and_coords

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from scipy import special
from sklearn.cluster import KMeans

class BGMM(Model):

    def __init__(self, data, num_gaussians, verbose=None, α0=None, m0=None, β0=None, W0=None, ν0=None):
        Model.__init__(self, data, verbose=verbose)
        self.num_gaussians = num_gaussians

        # hyper-parameters
        if α0 is not None:
            self.α0 = α0
        else:
            self.α0 = 1e-3

        if m0 is not None:
            self.m0 = m0
        else:
            self.m0 = np.zeros(shape=(self.num_features, ))
        
        if β0 is not None:
            self.β0 = β0
        else:
            self.β0 = 1

        if W0 is not None:
            self.W0 = W0
        else:
            self.W0 = np.eye(self.num_features)
        
        if ν0 is not None:
            self.ν0 = ν0
        else:
            self.ν0 = self.num_features

        self.rs = np.zeros(shape=(self.N, self.num_gaussians))

        # intial params
        # use k-means to initialise the responsibilities
        self.ms  = [np.zeros(shape=(self.num_features,))]*self.num_gaussians
        self.Ws = [np.zeros(shape=(self.num_features, self.num_features))]*self.num_gaussians
        mean_imputed_X = self.X.data.copy()
        mean_imputed_X[self.X.mask] = ma.mean(self.X, axis=0)[np.where(self.X.mask)[1]]
        kmeans = KMeans(n_clusters=self.num_gaussians, random_state=0).fit(mean_imputed_X)
        self.rs[np.arange(self.N), kmeans.labels_] = 1

        # for the rest of the params just use the priors
        self.αs = [self.α0]*self.num_gaussians
        self.βs = [self.β0]*self.num_gaussians
        self.νs = [self.ν0]*self.num_gaussians

        # to speed up convergence we can actually use the initial cluster assignemnts to determine what that params should be
        self._update_params()

        # get the initial ML estimate of the missing values and the corresponding LL
        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        if self.verbose: print("Fitting model:")

        for i in range(max_iters):
            old_αs, old_ms, old_βs, old_Ws, old_νs = self.αs, self.ms, self.βs, self.Ws, self.νs

            # do one iteration of inference
            self._calc_rs()
            self._update_params()

            # update the ML imputation and the LL
            self._calc_ML_est()
            self._calc_ll()
            
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))

            if np.linalg.norm(old_αs - np.array(self.αs)) + np.linalg.norm(old_ms - np.array(self.ms)) + np.linalg.norm(old_βs - np.array(self.βs)) \
                + np.linalg.norm(old_Ws - np.array(self.Ws)) + np.linalg.norm(old_νs - np.array(self.νs)) <= ϵ:
                break

    # "E-step"
    def _calc_rs(self):
        # E[log(π)]
        log_πs = special.digamma(self.αs) - special.digamma(np.sum(self.αs))
        log_rs = np.stack([log_πs]*self.N, axis=0)

        # E[log(N_nk)]
        for n in range(self.N):
            mask_row = self.X[n, :].mask

            o_locs, _, oo_coords, _, _, _ = get_locs_and_coords(mask_row)
            
            if not o_locs.size: continue

            x_row = self.X[n,:].data
            log_rs[n, :] += 0.5*np.log(np.array([linalg.det(self.Ws[k][oo_coords].reshape(o_locs.size, o_locs.size)) for k in range(self.num_gaussians)]))

            for k in range(self.num_gaussians):
                log_rs[n, k] += 0.5*np.sum([special.digamma(0.5*(self.νs[k] - self.num_features + o_locs.size + 1 - i)) for i in range(o_locs.size)]) 

            for k in range(self.num_gaussians):
                tmp = 0.5*self.βs[k]
                tmp *= (self.νs[k] - self.num_features + o_locs.size)
                tmp *= (x_row[o_locs] - self.ms[k][o_locs]).T
                tmp = tmp @ self.Ws[k][oo_coords].reshape(o_locs.size, o_locs.size)
                tmp = tmp @ (x_row[o_locs] - self.ms[k][o_locs])
                log_rs[n, k] -= tmp

        rs = np.exp(log_rs) + 1e-9 # incase none of the components want to take charge of an example
        self.rs = rs/np.sum(rs, axis=1, keepdims=True)        

    # "M-step"
    def _update_params(self):
        Ns = np.sum(self.rs, axis=0)

        self.αs = [Ns[k] + self.α0 for k in range(self.num_gaussians)]

        prev_ms = self.ms.copy()
        
        for k in range(self.num_gaussians):
            # check if the component is still valid
            if Ns[k] == 0: continue

            # build a repaired version of the data to use for the subsequent calculations
            x_rep = self.X.data.copy()
            for n in range(self.N):
                mask_row = self.X[n, :].mask

                _, m_locs, _, _, _, _ = get_locs_and_coords(mask_row)
                x_rep[n, m_locs] = prev_ms[k][m_locs]

            x_bar = 1/Ns[k]*np.sum(self.rs[:, k, np.newaxis] * x_rep, axis=0)

            self.βs[k] = self.β0 + Ns[k]
            self.νs[k] = self.ν0 + Ns[k]
            self.ms[k] = 1/self.βs[k]*(self.β0*self.m0 + Ns[k]*x_bar)
            W_inv = linalg.inv(self.W0)
            W_inv += np.einsum('ij,ikl->kl', self.rs[:, k, np.newaxis], np.einsum('ij,ik->ijk' ,x_rep - x_bar, x_rep - x_bar))
            W_inv += self.β0*Ns[k]/(self.β0 + Ns[k])*np.outer(x_bar - self.m0, x_bar - self.m0)
            self.Ws[k] = np.linalg.inv(W_inv)

    def _calc_ML_est(self):
        # Note: the expectation for each mean is simply self.ms[k],
        # similarly, the expectation for the precision is self.νs[k]*self.Ws[k]
        self.expected_X = self.X.data

        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs, m_locs, _, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

            k = np.argmax(self.rs[n, :])
            μ_k = self.ms[k]
            Λ_k = self.νs[k]*self.Ws[k]
            self.expected_X[n, m_locs] = μ_k[m_locs]

            # if there were any observations we can use that infomation to further update our estimate
            if o_locs.size:
                Σ = np.linalg.inv(Λ_k[mm_coords].reshape(m_locs.size, m_locs.size))
                self.expected_X[n, m_locs] -= Σ @ Λ_k[mo_coords].reshape(m_locs.size, o_locs.size) @ (x_row[o_locs] - μ_k[o_locs])
    
    def _calc_ll(self):
        lls = []
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue

            o_locs, m_locs, _, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

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

            lls.append(np.log(tmp))
        self.ll = np.mean(lls)

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for n in range(self.N):
            # figure out the conditional distribution for the missing data given the observed data
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask
            # if there are no missing values then go to next iter
            if np.all(~mask_row): continue

            # figure out which values are missing
            o_locs, m_locs, _, mm_coords, mo_coords, _ = get_locs_and_coords(mask_row)

            for i in range(num_samples):
                # sample from the multinomial to determine which gaussian to sample from
                choice = np.random.choice(self.num_gaussians, p=self.rs[n, :])
                # now sample the precision and mean from that gaussian
                Λ = stats.wishart.rvs(df=self.νs[choice], scale=self.Ws[choice])
                μ = stats.multivariate_normal.rvs(mean=self.ms[choice], cov=linalg.inv(self.βs[choice]*Λ))

                var = np.linalg.inv(Λ[mm_coords].reshape(m_locs.size, m_locs.size))
                mean = μ[m_locs]
                if o_locs.size:
                    mean -= var @ Λ[mo_coords].reshape(m_locs.size, o_locs.size) @ (x_row[o_locs] - μ[o_locs])

                sampled_Xs[i, n, m_locs] = stats.multivariate_normal.rvs(mean=mean, cov=var, size=1)

        return sampled_Xs
