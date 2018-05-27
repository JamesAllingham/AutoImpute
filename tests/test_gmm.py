# James Allingham
# April 2018
# test_gmm.py
# Tests for the GMM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from gmm import GMM
import testing_utils


class NoMissingValuesRMSETestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)
        model.fit()

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesCompleteInitialLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        ll = model.log_likelihood(return_mean=True, complete=True)

        self.assertAlmostEqual(ll, -3.179507)

class NoMissingValuesCompleteFinalLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        ll1 = model.log_likelihood(return_mean=True, complete=True)
        model.fit()
        ll2 = model.log_likelihood(return_mean=True, complete=True)

        self.assertAlmostEqual(ll1, ll2)

class NoMissingValuesMissingLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        ll = model.log_likelihood(return_mean=True, complete=False)

        self.assertTrue(np.isnan(ll))

class AllMissingValuesMLEResultTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        model.fit()
        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,3)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class AllMissingValuesMultipleComponenetsMLEResultTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=3, verbose=False, map_est=False)

        model.fit()
        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,3)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):

    def runTest(self):

        model = GMM(self.data, 1, verbose=False, independent_vars=True, map_est=False)
        
        model.fit()
        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,1)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneValueMLEResultTestCase(testing_utils.OneValueBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        model.fit()
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 2, 3]*3).reshape(3,3)))

class TwoValueMLEResultTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        model.fit()
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 3, 5, 6, 4, 2, 3.5, 3.5, 3.5]).reshape(3,3)))

class TwoValuesSamplesDifferentTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        model.fit()
        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class IndependentVsDependentLLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):

        model_ind = GMM(self.data, 1, verbose=False, independent_vars=True, map_est=False)
        model_ind.fit()
        ll_ind = model_ind.log_likelihood(complete=False, return_mean=True)

        model_dep = GMM(self.data, 1, verbose=False, independent_vars=False, map_est=False)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertGreater(ll_dep, ll_ind)

class OneColumnPredTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = GMM(self.data, num_components=1, verbose=False, map_est=False)
        model.fit()

        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([[1], [2], [1.5]])))

class OneColumnSampleTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = GMM(self.data, num_components=1, verbose=False, map_est=False)
        model.fit()

        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class OneColumnLLTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = GMM(self.data, 1, verbose=False, independent_vars=False)
        model.fit()

        model_dep = GMM(self.data, 1, verbose=False, independent_vars=False)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertEqual(ll_dep, model.log_likelihood(complete=False, return_mean=True))

class TwoCompLLSmallerThan10CompTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):

        model2 = GMM(self.data, 2, verbose=False, independent_vars=True, map_est=False)
        model2.fit()
        ll2 = model2.log_likelihood(complete=False, return_mean=True)

        model10 = GMM(self.data, 10, verbose=False, independent_vars=True, map_est=False)
        model10.fit()
        ll10 = model10.log_likelihood(complete=False, return_mean=True)
        
        self.assertGreater(ll10, ll2)

class MAPandMLEGiveDifferentLLsTestCase(testing_utils.IrisMCAR20BaseTestCase):

    def runTest(self):

        modelMLE = GMM(self.data, 2, verbose=False, independent_vars=True, map_est=False)
        modelMLE.fit()
        llMLE = modelMLE.log_likelihood(complete=False, return_mean=True)

        modelMAP = GMM(self.data, 2, verbose=False, independent_vars=True, map_est=True)
        modelMAP.fit()
        llMAP = modelMAP.log_likelihood(complete=False, return_mean=True)

        self.assertNotAlmostEqual(llMLE, llMAP)

class TenCompDoesntCrashOnBoston10TestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):

        model = GMM(self.data, 10, verbose=False, independent_vars=True, map_est=False)
        raised = False
        try:
            model.fit()
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')

class TenCompMAPDoesntCrashOnIris50TestCase(testing_utils.IrisMCAR50BaseTestCase):

    def runTest(self):

        model = GMM(self.data, 10, verbose=False, independent_vars=True, map_est=True)
        raised = False
        try:
            model.fit()
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')
