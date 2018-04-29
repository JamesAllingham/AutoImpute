# James Allingham
# April 2018
# mmm.py
# Imputation using a Mixed Mixture Model (consisiting of categorical and gaussian components) fitted using the EM algorithm
# Based on Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

# NOTE: scipy.stats.multinomial sometimes eroneously gives nan for Scipy versions lower than 1.10.0

from model import Model
from utilities import get_locs_and_coords, encode_1_hot, regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy import linalg
from sklearn.cluster import KMeans

class MMM(Model):

    def __init__(self, data, num_components, assignments=None, verbose=None):
        Model.__init__(self, data, verbose=verbose)

        self.num_components = num_components
        
        self._calc_ML_est()
        self._calc_ll()

        # get list of unique values in each column
        self.unique_vals = [np.unique(data[:, d].compressed()) for d in range(self.num_features)]

        # create a dictionary from value to one-hot encoding of which unique value it is
        self.one_hot_lookups = [{val: encode_1_hot(i, len(self.unique_vals[d])) 
            for i, val in enumerate(self.unique_vals[d])}
                for d in range(self.num_features)]

        # check if assignments were made and if so whether or not they were valid
        if assignments is None:
            assignments = ['r']*self.num_features
        elif len(assignments) != self.num_features:
            raise RuntimeError("Only %s assignemnt(s) were given. Please give one assignemnt per column (%s assignment(s))" % (len(assignments), self.num_features))
        
        for d, assignment in enumerate(assignments):
            if assignment != 'r' or assignment != 'd':
                raise RuntimeError("Invalid assignment (%s) given for column %s. Use 'r' and 'd' for real and discrete valued columns respectively." % (assignment, d))
        
        self.real_columns = assignments == 'r'

        # use k-means to initialise params
        self.rs = np.zeros(shape=(self.N, self.num_components))
        mean_imputed_X = self.X.data.copy()
        mean_imputed_X[self.X.mask] = ma.mean(self.X, axis=0)[np.where(self.X.mask)[1]]
        kmeans = KMeans(n_clusters=self.num_components, random_state=0).fit(mean_imputed_X)
        self.rs[np.arange(self.N), kmeans.labels_] = 1

        # create parameters depending on the feature assingments
        self.μs = np.stack([np.mean(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0) for k in range(self.num_components)], axis=0)
        self.Σs = np.stack([regularise_Σ(np.diag(np.var(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0))) for k in range(self.num_components)], axis=0)
        self.ps = np.array([[np.mean([self.one_hot_lookups[d][self.X.data[n, d]] for n in self.N if self.X.mask[n, d]], axis=0) 
            for d in range(self.num_features)] for k in range(self.num_components)])

        # randomise initial responsibilities
        self.rs = np.random.dirichlet(np.ones((self.N,)), self.num_components).T

        self._calc_ML_est()
        self._calc_ll()


    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μs, old_Σs, old_ps, old_rs, old_expected_X = self.μs.copy(), self.Σs.copy(), self.ps.copy(), self.rs.copy(), self.expected_X.copy()

            # E-step
            self._calc_rs()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll < best_ll or self.ll - best_ll < ϵ:
                self.μs, self.Σs, self.ps, self.rs, self.expected_X = old_μs, old_Σs, old_ps, old_rs, old_expected_X
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
            
            if not np.all(mask_row):
                o_r_locs = np.where(np.logical_and(~mask_row, self.real_columns))[0] 
                o_d_locs = np.where(np.logical_and(~mask_row, ~self.real_columns))[0] 
                
                rs[n, :] = np.ones((self.num_components,))

                # if there are any observed real vars in this row then include them in the responsibility
                sz = len(o_r_locs)
                if sz:
                    oo_r_coords = tuple(zip(*[(i, j) for i in o_r_locs for j in o_r_locs]))
                    rs[n, :] *= np.array([stats.multivariate_normal.pdf(x_row[o_r_locs], mean=self.μs[k, o_r_locs], cov=self.Σs[k, :, :][oo_r_coords].reshape(sz, sz), allow_singular=True)
                        for k in range(self.num_components)])
                # if there are any observed discrete vars then include them
                if len(o_d_locs):
                    rs[n, :] *= np.array([np.prod([stats.multinomial.pmf(self.one_hot_lookups[d][x_row[o_d_locs][d]], 1, self.ps[d][:, k]) for d in o_d_locs])
                        for k in range(self.num_components)])
            else:
                rs[n, :] = np.mean(self.rs, axis=0)

    # M-step
    def _update_params(self):
        # update the ps
        ps = np.array([np.zeros(shape=(self.unique_vals[d].size, self.num_components)) for d in range(self.num_features)])

        for d, is_real in enumerate(self.real_columns):
            if is_real: continue
            
            for k in range(self.num_components):
                
                tmp = 0
                for n in range(self.N):
                    tmp += self.rs[n, k]*(self.one_hot_lookups[d][self.X.data[n, d]] if not self.X.mask[n, d] else self.ps[d][:, k])

                tmp /= np.sum(self.rs[:, k])
                ps[d][:, k] = tmp

        self.ps = ps

        for k in range(self.num_components):
            # update the μs
            # first fill in the missing elements of X
            X = self.X.data
            μ = np.stack([self.μs[k]]*self.N, axis=0)
            X[self.X.mask] = μ[self.X.mask] # TODO: make this work when dealing with full cov matrices

            p = self.rs[:, k]
            self.μs[k] = (p @ X)/np.sum(p)

            # now do Σs
            C = np.zeros(shape=(self.num_features, self.num_features))
            for n in range(self.N):
                mask_row = self.X[n, :].mask
                C[np.ix_(mask_row, mask_row)] += self.Σs[k][np.ix_(mask_row, mask_row)] # TODO: make this work with full cov matrices
            diff = (X - self.μs[k])*np.sqrt(p)
            self.Σs[k] = np.diag(np.sum(diff*diff, axis=0)/np.sum(p)) # Note this will need to change to an outer product for full cov something like np.einsum('ij,ik->ijk', foo, foo)

            self.Σs[k] += C
            self.Σs[k] = regularise_Σ(self.Σs[k])

    def _calc_ML_est(self):
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue
            
            m_r_locs = np.where(np.logical_and(mask_row, self.real_columns))[0] 
            m_d_locs = np.where(np.logical_and(mask_row, ~self.real_columns))[0] 

            # impute real values
            if m_r_locs.size:
                self.expected_X[n, :][m_r_locs] = np.sum(self.rs[n, :]*self.μs[m_r_locs], axis=0) # TODO: make this work with full cov

            # impute discrete values
            for d in m_d_locs:
                self.expected_X[n, d] = self.unique_vals[d][np.argmax(np.sum(self.rs[n, :]*self.ps[d][:, :], axis=1))]

    def _calc_ll(self):
        lls = []

        for n in range(self.N):
            mask_row = self.X[n, :].mask
            if np.all(~mask_row): continue

            m_r_locs = np.where(np.logical_and(mask_row, self.real_columns))[0] 
            m_d_locs = np.where(np.logical_and(mask_row, ~self.real_columns))[0] 

            prob = 0
            for k in range(self.num_components):
                tmp = 1

                # probability of real values
                μmo = self.μs[k, m_r_locs] # TODO: make this work with full cov mat
                Σmm = self.Σs[k, :, :][np.ix_(m_r_locs, m_r_locs)]

                tmp *= self.rs[n, k] * stats.multivariate_normal.pdf(self.expected_X[n, m_r_locs], mean=μmo, cov=Σmm, allow_singular=True)

                # probability of discrete values
                for d in m_d_locs:
                    tmp *= self.rs[n, k]*stats.multinomial.pmf(self.one_hot_lookups[d][self.expected_X[n, d]] ,1, self.ps[d][:, k])

                prob += tmp

            lls.append(np.log(prob)) 

        self.ll = np.mean(lls)

    def sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):

            for n in range(self.N):
                mask_row = self.X[n, :].mask
                # if there are no missing values then go to next iter
                if np.all(~mask_row): continue

                m_r_locs = np.where(np.logical_and(mask_row, self.real_columns))[0] 
                m_d_locs = np.where(np.logical_and(mask_row, ~self.real_columns))[0] 
                k = np.random.choice(self.num_components, p=self.rs[n, :])
                
                # sample real values
                μmo = self.μs[k, m_r_locs] # TODO: make this work with full cov mat
                Σmm = self.Σs[k, :, :][np.ix_(m_r_locs, m_r_locs)]
                sampled_Xs[i, n, m_r_locs] = stats.multivariate_normal.rvs(mean=μmo, cov=Σmm, size=1)

                # sample discrete values
                for d in m_d_locs:
                    sampled_Xs[i, n, d] = self.unique_vals[d][np.argmax(stats.multinomial.rvs(1, self.ps[k, d]))]

        return sampled_Xs
        