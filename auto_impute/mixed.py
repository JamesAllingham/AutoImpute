# James Allingham
# May 2018
# mixed.py
# Imputation using DPs and GMMs

from model import Model
from dp import DP
from gmm import GMM

import numpy as np
from scipy import stats

class Mixed(Model):

    def __init__(self, data, verbose=None, assignments=None, num_components=3):
        Model.__init__(self, data, verbose=verbose, normalise=False)

        self.num_components = num_components

        # check if assignments were made and if so whether or not they were valid
        if assignments is not None:
            if len(assignments) != self.num_features:
                print("%s assignemnt(s) were given. Please give one assignemnt per column (%s assignment(s))" % (len(assignments), self.num_features))
                exit(1)

            for d, assignment in enumerate(assignments):                
                if assignment != 'c' and assignment != 'd':
                    print("Invalid assignment ('%s') given for column %s. Use 'c' and 'd' for continuous and discrete valued columns respectively." % (assignment, d))
                    exit(1)

            self.ts = np.array([1 if assignment == 'c' else 0 for assignment in assignments])
        else:
            self.ts = np.array([1]*self.num_features)

        # create a DP for each column marked 'd' and a GMM for each column marked 'c'
        self.models = []
        for i, t in enumerate(self.ts):
            if t == 1:
                model = GMM(data[:, i, np.newaxis], self.num_components, verbose=False, map_est=False)
                model.fit()
                self.models.append(model)
            else:
                model = DP(data[:, i, np.newaxis])
                self.models.append(model)

        self._calc_ML_est()
        self._calc_ll()

    def _calc_ML_est(self):
        ml_ests = []

        for model in self.models:
            ml_ests.append(model.impute().flatten())

        self.expected_X = np.stack(ml_ests, axis=1)

    def _calc_ll(self):
        self.lls = []

        for model in self.models:
            self.lls += model.log_likelihood(return_individual=True)

        self.ll = np.mean(self.lls)

    def _sample(self, num_samples):
        sampled_Xs = []

        for i in range(num_samples):
            X = []

            for model in self.models:
                X.append(model.sample(1).flatten())

            sampled_Xs.append(np.stack(X, axis=1))

        return np.stack(sampled_Xs, axis=0)

