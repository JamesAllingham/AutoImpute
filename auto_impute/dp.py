# James Allingham
# May 2018
# dp.py
# Imputation using a Dirichlet process

from model import Model

import numpy as np
from scipy import stats
import copy

class DP(Model):

    def __init__(self, data, verbose=None, normalise=False, α=1, G=None):
        Model.__init__(self, data, verbose=verbose, normalise=normalise)

        self.α = α
        if G is None:
            self.G = stats.norm(loc=0, scale=10000)
        else:
            self.G = G

        # for each column, create a map from unique value to number of occurances
        self.col_lookups = [
            {
                unique_val: count 
                for unique_val, count in zip(*np.unique(self.X.data[:,d][~self.X.mask[:,d]], return_counts=True))
            }
            for d in range(self.D)
        ]

        self._calc_ML_est()
        self._calc_ll()

    def _calc_ML_est(self):
        # copy so that we don't modify the original data
        col_lookups_ = copy.deepcopy(self.col_lookups)

        # complete each column going from top to bottom
        self.expected_X = self.X.data
        for d in range(self.D):
            for n in range(self.N):
                # skip this iter if we don't need to impute a value
                if not self.X.mask[n, d]:
                    continue

                # impute the current value
                # we do this by taking the most likely value from categorical distribution with num + 1 outcomes, 
                # where num is the number of unique observations for the column
                # the extra 1 comes from the chance to draw a new observation from G
                pvals = np.array(list(col_lookups_[d].values()) + [self.α])
                pvals = pvals/np.sum(pvals)
                choice = np.argmax(pvals)

                # if the choice was the new observation
                if choice == pvals.size - 1:
                    # then use the most likely value of the distribution
                    x = self.G.mean()
                    self.expected_X[n, d] = x
                    # and update col_lookups_
                    if x in col_lookups_[d]:
                        col_lookups_[d][x] += 1
                    else:
                        col_lookups_[d][x] = 1
                else:
                    # use the appropriate value
                    x = list(col_lookups_[d].keys())[choice]
                    self.expected_X[n, d] = x
                    # increase the approrpiate counter
                    col_lookups_[d][x] += 1
                    

    def _calc_ll(self):
        # copy so that we don't modify the original data
        col_lookups_ = [{ } for d in range(self.D)]
        
        for d in range(self.D):
            for n in range(self.N):
                x = self.X.data[n, d]

                N = np.sum(list(col_lookups_[d].values()))
                p_x = 0
                if x in col_lookups_[d]:
                    p_x += col_lookups_[d][x]/(N + self.α)  
                p_x += self.α/(N + self.α)*self.G.pdf(x)
                self.lls[n, d] = np.log(p_x)

                # update col_lookups_
                if x in col_lookups_[d]:
                    col_lookups_[d][x] += 1
                else:
                    col_lookups_[d][x] = 1

    def _sample(self, num_samples):
        sampled_Xs = np.stack([self.X.data]*num_samples, axis=0)

        for i in range(num_samples):
            # copy so that we don't modify the original data
            col_lookups_ = copy.deepcopy(self.col_lookups)

            # complete each column going from top to bottom
            for d in range(self.D):
                for n in range(self.N):
                    # skip this iter if we don't need to impute a value
                    if not self.X.mask[n, d]:
                        continue

                    # impute the current value
                    # we do this by drawing from a categorical distribution with num + 1 outcomes, 
                    # where num is the number of unique observations for the column
                    # the extra 1 comes from the chance to draw a new observation from G
                    pvals = np.array(list(col_lookups_[d].values()) + [self.α])
                    pvals = pvals/np.sum(pvals)
                    choice = np.argmax(np.random.multinomial(1, pvals))
                    # if d == self.D - 1: print(choice, pvals.size)

                    # if the choice was the new observation
                    if choice == pvals.size - 1:
                        # then use the most likely value of the distribution
                        x = self.G.rvs()
                        sampled_Xs[i, n, d] = x
                        # and update col_lookups_
                        if x in col_lookups_[d]:
                            col_lookups_[d][x] += 1
                        else:
                            col_lookups_[d][x] = 1
                    else:
                        # use the appropriate value
                        x = list(col_lookups_[d].keys())[choice]
                        sampled_Xs[i, n, d] = x
                        # increase the approrpiate counter
                        col_lookups_[d][x] += 1

        return sampled_Xs