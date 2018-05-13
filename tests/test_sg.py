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

from sg import SingleGaussian
import testing_utils

class NoMissingValuesRMSETestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        imputed_X = model.impute()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertGreaterEqual(rmse, 0.0)

class NoMissingValuesInitialLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertTrue(np.isnan(ll)) # def LL(sig, mu, k, x): return -0.5*(np.log(np.linalg.det(sig)) + (x - mu).T @ np.linalg.inv(sig) @ (x - mu) + k*np.log(2*np.pi) )

class BostonMCAR10LLTestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):        
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -1.3819994419383168)

class BostonMCAR20LLTestCase(testing_utils.BostonMCAR20BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -1.916096390907473)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -2.3689620433382492)

class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.9648759753923146)

class IrisMCAR20LLTestCase(testing_utils.IrisMCAR20BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.9668574559009642)

class IrisMCAR40LLTestCase(testing_utils.IrisMCAR40BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.9725021074844706)

class IrisMCAR50LLTestCase(testing_utils.IrisMCAR50BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.6286605534111728)
