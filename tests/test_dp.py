# James Allingham
# May 2018
# test_SingleGaussianEM.py
# Tests for the Dirichlet process 

import sys
import numpy as np
from scipy import stats

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from dp import DP
import testing_utils

class EandPiResultTestCase(testing_utils.EandPiBaseTestCase):
    """Tests the predictions of the maximum likelihood imputation of the DP using.
    """
    def runTest(self):

        model = DP(self.data, verbose=None, α=0.5, G=stats.norm(loc=0, scale=1))

        imputed_X = model.ml_imputation()

        expected_X = np.array(
            [
                [2.71828, 3.14159, 3.14159, 3.14159, 0],
                [2.71828, 2.71828, 2.71828, 3.14159, 0],
                [2.71828, 2.71828, 3.14159, 3.14159, 0],
                [2.71828, 2.71828, 2.71828, 3.14159, 0],
                [2.71828, 2.71828, 2.71828, 3.14159, 0],
                [2.71828, 3.14159, 3.14159, 3.14159, 0],
                [2.71828, 2.71828, 3.14159, 3.14159, 0],
                [2.71828, 2.71828, 3.14159, 3.14159, 0]
             ]
        )

        self.assertTrue(np.all(expected_X == imputed_X))

class EandPiLLTestCase(testing_utils.EandPiBaseTestCase):
    """Tests the missing data log-likelihoods of the maximum likelihood imputation of the DP.
    """
    def runTest(self):

        model = DP(self.data, verbose=False, α=0.5, G=stats.norm(loc=0, scale=1))

        lls = model.log_likelihood(complete=True, return_individual=True)

        expected_lls = np.array(
            [
                [-4.613461612404673, -5.853732397254673, -5.853732397254673, -5.853732397254673, -0.9189385332046727],
                [-0.40051864777600404, -5.712073901072783, -5.712073901072783, -0.40403155100979277, -0.22358436495767234],
                [-0.22066726271752038, -0.9113442715419946, -0.9148571747757833, -0.22242651587930157, -0.12807379122624513],
                [-0.15249913961646647, -0.5571394993387334, -1.2478165081632075, -0.15367259908189232, -0.08977744103533772],
                [-0.11654412485758293, -0.40381356789737255, -0.8084539276196395, -0.117424453671454, -0.06911879841524382],
                [-0.09431892840371207, -1.7033145351400534, -1.0108838762435717, -0.09502330393063575, -0.056191175630781125],
                [-0.07921659662004565, -0.4842689049829001, -0.7727118074881156, -0.07980363873070372, -0.047338188456421064],
                [-0.06828473452322996, -0.4044738567075516, -0.6282500774374447, -0.06878795175121491, -0.04089544996794101]
            ]
        )

        self.assertTrue(np.all(lls == expected_lls))

class NotAllZerosGivenNoObsTestCase(testing_utils.EandPiBaseTestCase):
    """Tests that the sample function works when given only missing data.
    """
    def runTest(self):

        model = DP(self.data, verbose=False)

        result = model._sample(1)[0]

        self.assertFalse(np.all(result[:, 4] == 0))

class SingleColumnSampleTest(testing_utils.OneColumnBaseTestCase):
    """Tests that a) the DP works on a single column, and that b) if there are no missing values the output will be the observed data.
    """
    def runTest(self):

        model = DP(self.data, verbose=False)

        sampled_X = model.sample(1)

        self.assertTrue(np.all(sampled_X[0, 0:2, :] == np.array([[1], [2]])))

class SingleColumnMultipleSample(testing_utils.OneColumnBaseTestCase):
    """Tests that all samples from a dataset with no missing values will be the same.
    """
    def runTest(self):

        model = DP(self.data, verbose=False)

        sampled_X = model.sample(100)

        self.assertFalse(np.all(sampled_X[:, 2, :] == sampled_X[0,2,0]))

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):
    """Tests that the maximum likelihood prediction of the DP on a dataset with only missing values is the mode of the base distribution G_0 .
    """
    def runTest(self):

        model = DP(self.data, verbose=False)
        
        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,1)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)
