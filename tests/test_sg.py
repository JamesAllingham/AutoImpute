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

        self.assertAlmostEqual(rmse, 0.0)

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

        self.assertAlmostEqual(ll, -4.153074650106323)

class BostonMCAR20LLTestCase(testing_utils.BostonMCAR20BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -6.585943797329661)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -8.730701203328083)

class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.6722662455291954)

class IrisMCAR20LLTestCase(testing_utils.IrisMCAR20BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.6831221650724867)

class IrisMCAR40LLTestCase(testing_utils.IrisMCAR40BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.6206492288167005)

class IrisMCAR50LLTestCase(testing_utils.IrisMCAR50BaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.1454619352139332)
