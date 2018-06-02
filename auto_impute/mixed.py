# James Allingham
# May 2018
# mixed.py
# Imputation using DPs and GMMs

from model import Model
from dp import DP
from gmm import GMM
from sg import SingleGaussian
from utilities import print_err

import numpy as np
from scipy import stats

class Mixed(Model):

    def __init__(self, data, verbose=None, assignments=None, num_components=3, α0=None, m0=None,
        β0=None, W0=None, ν0=None, α=None, G=None):
        Model.__init__(self, data, verbose=verbose)

        self.num_components = num_components
        self.assignments_given = False

        # hyper-parameters
        if α0 is not None:
            α0 = α0
        else:
            α0 = 1

        if m0 is not None:
            m0 = m0
        else:
            m0 = np.zeros(shape=(1, ))
        
        if β0 is not None:
            β0 = β0
        else:
            β0 = 1e-0

        if W0 is not None:
            W0 = W0
        else:
            W0 = np.eye(1)
        
        if ν0 is not None:
            ν0 = ν0
        else:
            ν0 = 1
        
        if α is not None:
            α = α
        else:
            α = 0.5
        
        if G is not None:
            G = G
        else:
            G = stats.norm(loc=0, scale=10000)

        # check if assignments were made and if so whether or not they were valid
        if assignments is not None:
            if len(assignments) != self.D:
                print_err("%s assignemnt(s) were given. Please give one assignemnt per column (%s assignment(s))" % (len(assignments), self.D))
                exit(1)

            for d, assignment in enumerate(assignments):                
                if assignment != 'c' and assignment != 'd':
                    print_err("Invalid assignment ('%s') given for column %s. Use 'c' and 'd' for continuous and discrete valued columns respectively." % (assignment, d))
                    exit(1)

            self.ts = np.array([0 if assignment == 'c' else 1 for assignment in assignments])
            self.assignments_given = True
        else:
            self.ts = np.ones(shape=(self.D, 2))
            self.ts /= np.sum(self.ts, axis=1, keepdims=True)

        # self.ts = np.array([encode_1_hot(t, 2) for t in self.ts])
        self.πs = np.array([np.array([1/2]*2)]*self.D)

        # create a both a DP and a GMM for each column
        self.models = [[], []]
        for d in range(self.D):
            # GMM
            # model = GMM(data[:, d, np.newaxis], self.num_components, verbose=False, map_est=True, α0=α0, m0=m0, β0=β0, W0=W0, ν0=ν0)
            # model.fit()
            # self.models[0].append(model)
            self.models[0].append(SingleGaussian(data[:, d, np.newaxis], verbose=False, map_est=True, m0=np.ones_like(m0)*0, β0=β0, W0=W0, ν0=ν0))
            self.models[0][d].fit()
            # DP
            # self.models[1].append(DP(data[:, d, np.newaxis], α=0.001, G=stats.norm(loc=0, scale=1000)))
            self.models[1].append(SingleGaussian(data[:, d, np.newaxis], verbose=False, map_est=True, m0=np.ones_like(m0)*100, β0=β0, W0=W0, ν0=ν0))
            self.models[1][d].fit()

        # self._e_step()
        τ = np.zeros_like(self.ts)
        for d in range(self.D):
            for i in range(τ.shape[1]):
                π = self.πs[d, i]
                τ[d, i] = π*self.models[i][d].evidence()

        print(τ)

        self.ts = τ / np.sum(τ, axis=1, keepdims=True)

        self._calc_ML_est()
        self._calc_ll() 

    def _continuous_probs(self):
        return self.ts[:,0]   

    def _e_step(self):
        τ = np.zeros_like(self.ts)
        for d in range(self.D):
            for i in range(τ.shape[1]):
                π = self.πs[d, i]
                τ[d, i] = π*np.exp(self.models[i][d].log_likelihood(observed=True))
            
        # deal with edge case where there are no observed values:
        τ[np.isnan(τ)] = 1e-64
        self.ts = τ / np.sum(τ, axis=1, keepdims=True)

    def _calc_ML_est(self):
        self.expected_X = np.zeros_like(self.X.data)

        for d in range(self.D):
            i = np.argmax(self.ts[d])
            self.expected_X[:, d] = self.models[i][d].ml_imputation().flatten()

    def _calc_ll(self):
        self.lls = np.zeros_like(self.X.data)
        for d in range(self.D):
            for i, t in enumerate(self.ts[d]): 
                self.lls[:, d] += t*self.models[i][d].log_likelihood(return_individual=True, complete=True).flatten()

    def _sample(self, num_samples):
        sampled_Xs = np.zeros(shape=(num_samples, ) + self.X.data.shape)

        for j in range(num_samples):
            for d in range(self.D):
                i = np.argmax(np.random.multinomial(1, self.ts[d]))
            
                sampled_Xs[j, :, d] = self.models[i][d].sample(1).flatten()

        return sampled_Xs

