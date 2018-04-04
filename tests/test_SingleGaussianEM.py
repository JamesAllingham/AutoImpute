# James Allingham
# April 2018
# test_SingleGaussianEM.py
# Tests for the single Gaussian EM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from SingleGaussianEM import SingleGaussian
from test_utils import NoMissingValuesBaseTestCase

class NoMissingValuesRMSETestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        imputed_X = model.impute()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesLLTestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -8.453194840524363)
