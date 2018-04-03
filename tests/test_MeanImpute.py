# James Allingham
# April 2018
# test_MeanImpute.py
# Tests for the mean imputation module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from MeanImpute import MeanImpute

class NoMissingValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        # create a fake data file to impute
        self.data = np.array([[1,2,3],
                         [4,5,6],
                         [7,8,9]])

class NoMissingValuesRMSETestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.impute()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEquals(rmse, 0.0)

class NoMissingValuesLLTestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, -np.inf)