# James Allingham
# April 2018
# mmm.py
# Imputation using a Mixed Mixture Model (consisiting of categorical and gaussian components) fitted using the EM algorithm
# Based on Ghahramani, Z., & Jordan, M. I. (n.d.). Supervised learning from incomplete data via an EM approach. 
# Retrieved from http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach.pdf

# NOTE: scipy.stats.multinomial sometimes eroneously gives nan for Scipy versions lower than 1.10.0

from model import Model
from utilities import encode_1_hot, regularise_Σ

import numpy as np
import numpy.ma as ma
from scipy import stats
from sklearn.cluster import KMeans

class MMM(Model):

    def __init__(self, data, num_components, assignments=None, verbose=None):
        Model.__init__(self, data, verbose=verbose)

        self.num_components = num_components

        # get list of unique values in each column
        self.unique_vals = [np.unique(self.X[:, d].compressed()) for d in range(self.num_features)]

        # create a dictionary from value to corresponding one-hot encoding
        self.one_hot_lookups = [{val: encode_1_hot(i, len(self.unique_vals[d])) 
            for i, val in enumerate(self.unique_vals[d])}
                for d in range(self.num_features)]

        # check if assignments were made and if so whether or not they were valid
        self.αssignments_given = False 
        if assignments != "":
            if len(assignments) != self.num_features:
                raise RuntimeError("%s assignemnt(s) were given. Please give one assignemnt per column (%s assignment(s))" % (len(assignments), self.num_features))

            for d, assignment in enumerate(assignments):
                if assignment != 'r' and assignment != 'd':
                    raise RuntimeError("Invalid assignment ('%s') given for column %s. Use 'r' and 'd' for real and discrete valued columns respectively." % (assignment, d))

            self.αssignments_given = True    
               

        # use k-means to initialise gaussian params
        mean_imputed_X = self.X.data.copy()
        mean_imputed_X[self.X.mask] = ma.mean(self.X, axis=0)[np.where(self.X.mask)[1]]
        kmeans = KMeans(n_clusters=self.num_components, random_state=0).fit(mean_imputed_X)

        # create parameters depending on the feature assingments
        rs = np.zeros(shape=(self.N, self.num_components))
        rs[np.arange(self.N), kmeans.labels_] = 1
        self.πs = np.mean(rs, axis=0)
        self.μs = np.stack([np.mean(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0) for k in range(self.num_components)], axis=0)
        self.Σs = np.stack([regularise_Σ(np.diag(np.var(mean_imputed_X[np.where(kmeans.labels_ == k)[0], :], axis=0))) for k in range(self.num_components)], axis=0)
        self.ps = np.array([[np.mean([self.one_hot_lookups[d][self.X.data[n, d]] for n in range(self.N) if ~self.X.mask[n, d]], axis=0) 
            for d in range(self.num_features)] for k in range(self.num_components)])
        # randomly init the catagorical params
        # self.ps = np.array([[np.random.dirichlet(np.ones(len(self.unique_vals[d]))) 
            # for d in range(self.num_features)] for k in range(self.num_components)])

        # randomise initial responsibilities
        self.rs = np.random.dirichlet(np.ones((self.N,))*10, self.num_components).T

        # for the column types, if they havent been given then randomly choose them
        if self.αssignments_given:
            self.αs = np.array([1 if assignment == 'r' else 0 for assignment in assignments])
        else:
            self.αs = np.array([0.5]*self.num_features) #np.random.rand(self.num_features)

        # use the same values for the initial column type priors
        self.ρs = self.αs

        self._calc_ML_est()
        self._calc_ll()

    def fit(self, max_iters=100, ϵ=1e-1):
        best_ll = self.ll
        if self.verbose: print("Fitting model:")
        for i in range(max_iters):
            old_μs, old_Σs, old_ps, old_πs, old_rs, old_expected_X, old_αs, old_ρs = self.μs.copy(), self.Σs.copy(), self.ps.copy(), self.πs.copy(), self.rs.copy(), self.expected_X.copy(), self.αs.copy(), self.ρs.copy()
            # E-step
            self._update_latent()

            # M-step
            self._update_params()
            
            self._calc_ML_est()
            # if the log likelihood stops improving then stop iterating
            self._calc_ll()
            if self.ll - best_ll < ϵ:
                self.μs, self.Σs, self.ps, self.πs, self.rs, self.expected_X, self.αs, self.ρs = old_μs, old_Σs, old_ps, old_πs, old_rs, old_expected_X, old_αs, old_ρs
                self.ll = best_ll

                if self.verbose and not self.αssignments_given:
                    print("\nFinal assignments (probability for real): " + " ".join(["%f" % x for x in self.αs]))

                break
            
            best_ll = self.ll
            if self.verbose: print("Iter: %s\t\tLL: %f" % (i, self.ll))
        print("")

    # E-step
    def _update_latent(self):
        rs = np.zeros(shape=(self.N, self.num_components))
        
        for n in range(self.N):
            x_row = self.X[n, :].data
            mask_row = self.X[n, :].mask
            o_locs = np.where(~mask_row)[0]
            
            if not np.all(mask_row):  
                rs[n, :] += self.πs*np.array([
                    np.prod([
                        self.ρs[d]*stats.norm.pdf(x_row[d], loc=self.μs[k][d], scale=np.diag(self.Σs[k])[d]) +
                        (1 - self.ρs[d])*stats.multinomial.pmf(self.one_hot_lookups[d][x_row[d]], 1, self.ps[k, d])
                        for d in o_locs
                    ])
                    for k in range(self.num_components)
                ])
                if np.all(rs[n, :] == 0): # deal with the case where no components want to take charge of an example
                    rs[n, :] = 1e-64
            else:
                rs[n, :] = self.πs
                
        self.rs = rs/np.sum(rs, axis=1, keepdims=True)

        if not self.αssignments_given:
            with np.errstate(over='ignore'):
                αs = self.ρs**1/((1 - self.ρs)**1 + 1e-64)*np.array([
                    np.prod([
                        np.sum([
                            self.πs[k]*stats.norm.pdf(self.X.data[n, d], loc=self.μs[k][d], scale=np.diag(self.Σs[k])[d]) for k in range(self.num_components)
                        ])/
                        np.sum([
                            self.πs[k]*stats.multinomial.pmf(self.one_hot_lookups[d][self.X.data[n, d]], 1, self.ps[k, d]) for k in range(self.num_components)
                        ])
                        for n in range(self.N) if not self.X.mask[n, d]
                    ])
                    for d in range(self.num_features)
                ])
                αs[np.isinf(αs)] = np.finfo(np.float64).max
            self.αs = αs/(αs + 1)

    # M-step
    def _update_params(self):
        # update πs
        self.πs = np.mean(self.rs, axis=0)

        # update ρs        
        self.ρs = self.αs

        # update the discrete params
        ps = np.array([[np.zeros(shape=(self.unique_vals[d].size)) for d in range(self.num_features)] for k in range(self.num_components)])

        for d in range(self.num_features):            
            for k in range(self.num_components):
                
                tmp = 0
                for n in range(self.N):
                    tmp += self.rs[n, k]*(self.one_hot_lookups[d][self.X.data[n, d]] if not self.X.mask[n, d] else self.ps[k, d])

                tmp /= np.sum(self.rs[:, k])
                ps[k, d] = tmp

        self.ps = ps

        # update the real params
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
                C[np.ix_(mask_row, mask_row)] += p[n]*self.Σs[k][np.ix_(mask_row, mask_row)]/np.sum(p) # TODO: make this work with full cov matrices
            
            diff = (X - self.μs[k]).T*np.sqrt(p)
            self.Σs[k] = np.diag(np.sum(diff*diff, axis=1)/np.sum(p)) # Note this will need to change to an outer product for full cov something like np.einsum('ij,ik->ijk', foo, foo)

            self.Σs[k] += C
            self.Σs[k] = regularise_Σ(self.Σs[k])

    def _calc_ML_est(self):
        self.expected_X = self.X.data.copy()
        for n in range(self.N):
            mask_row = self.X[n, :].mask

            if np.all(~mask_row): continue
            m_r_locs = np.where(np.logical_and(mask_row, self.αs >= 0.5))[0] 
            m_d_locs = np.where(np.logical_and(mask_row, np.logical_not(self.αs >= 0.5)))[0]

            # determine which cluster to use
            k = np.argmax(self.rs[n, :])
            # impute real values
            if m_r_locs.size:
                self.expected_X[n, m_r_locs] = self.μs[k, m_r_locs] # TODO: make this work with full cov

            # impute discrete values
            for d in m_d_locs:
                self.expected_X[n, d] = self.unique_vals[d][np.argmax(self.ps[k, d])]

    def _calc_ll(self):
        lls = []

        for n in range(self.N):
            mask_row = self.X[n, :].mask
            if np.all(~mask_row): continue

            m_locs = np.where(mask_row)[0]

            prob = 0
            for k in range(self.num_components):
                tmp = 1

                for d in m_locs:
                    # real part
                    r_tmp = self.αs[d] * stats.norm.pdf(self.expected_X[n, d], loc=self.μs[k, d], scale=self.Σs[k, d, d])

                    # disc part
                    try:
                        d_tmp = (1 - self.αs[d]) * stats.multinomial.pmf(self.one_hot_lookups[d][self.expected_X[n, d]], 1, self.ps[k, d])
                    except KeyError as _: # if the value cannot be looked up then its probabiltiy is 0 acroding the categorical distribution
                        d_tmp = 0

                    tmp *= r_tmp + d_tmp

                prob += self.rs[n, k]*tmp

            lls.append(np.log(prob)) 

        self.ll = np.mean(lls)

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):

            for n in range(self.N):
                mask_row = self.X[n, :].mask
                # if there are no missing values then go to next iter
                if np.all(~mask_row): continue

                m_locs = np.where(mask_row)[0]
                k = np.random.choice(self.num_components, p=self.rs[n, :])
                
                for d in m_locs:
                    # sample to determine if real or not
                    if np.random.rand(1) <= self.αs[d]:
                        # sample real value
                        sampled_Xs[i, n, d] = stats.norm.rvs(loc=self.μs[k, d], scale=self.Σs[k, d, d])
                    else:
                        # sample discrete value
                        sampled_Xs[i, n, d] = self.unique_vals[d][np.argmax(stats.multinomial.rvs(1, self.ps[k, d]))]

        return sampled_Xs
        