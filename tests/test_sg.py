# James Allingham
# April 2018
# test_sg.py
# Tests for the single Gaussian fitted with EM

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

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesCompleteInitialLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, normalise=False, independent_vars=True)

        ll = model.log_likelihood(return_mean=True, complete=True)

        self.assertAlmostEqual(ll, -3.179507)

class NoMissingValuesCompleteFinalLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, normalise=False)

        ll1 = model.log_likelihood(return_mean=True, complete=True)
        model.fit()
        ll2 = model.log_likelihood(return_mean=True, complete=True)

        self.assertAlmostEqual(ll1, ll2)

class NoMissingValuesMissingLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, normalise=False)

        ll = model.log_likelihood(return_mean=True, complete=False)

        self.assertTrue(np.isnan(ll))

class AllMissingValuesMLEResultTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, normalise=False)

        model.fit()
        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,3)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneValueMLEResultTestCase(testing_utils.OneValueBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)

        model.fit()
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 2, 3]*3).reshape(3,3)))

class TwoValueMLEResultTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False)

        model.fit()
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 3, 5, 6, 4, 2, 3.5, 3.5, 3.5]).reshape(3,3)))

class TwoValuesSamplesDifferentTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, normalise=True)

        model.fit()
        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class IndependentVsDependentLLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):

        model_ind = SingleGaussian(self.data, verbose=False, independent_vars=True)
        model_ind.fit()
        ll_ind = model_ind.log_likelihood(complete=False, return_mean=True)

        model_dep = SingleGaussian(self.data, verbose=False, independent_vars=False)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertGreater(ll_dep, ll_ind)

class OneColumnPredTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False)
        model.fit()

        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([[1], [2], [1.5]])))

class OneColumnSampleTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False)
        model.fit()

        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class OneColumnLLTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False)
        model.fit()

        model_dep = SingleGaussian(self.data, verbose=False, independent_vars=False)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertEqual(ll_dep, model.log_likelihood(complete=False, return_mean=True))
