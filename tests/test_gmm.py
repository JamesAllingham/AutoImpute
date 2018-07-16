# James Allingham
# April 2018
# test_gmm.py
# Tests for the GMM module

import unittest
import sys
import numpy as np
import numpy.ma as ma

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from gmm import GMM
import testing_utils


class NoMissingValuesRMSETestCase(testing_utils.NoMissingValuesBaseTestCase):
    """Tests that if there are no missing values the prediction RMSE is 0.
    """
    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)
        model.fit()

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesMissingLLTestCase(testing_utils.NoMissingValuesBaseTestCase):
    """Tests that the average missing data log-likelihood is not a number when there is no missing data.
    """
    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        ll = model.log_likelihood(return_mean=True, complete=False)

        self.assertTrue(np.isnan(ll))

class AllMissingValuesMLEResultDoesntChangeTestCase(testing_utils.AllMissingBaseTestCase):
    """Tests that the maximum likelihood predictions, before and after fitting a single component GMM with MLE, are the same if all the data is missing.
    """
    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        imputed_X1 = model.ml_imputation()

        model.fit()
        imputed_X2 = model.ml_imputation()

        rmse = np.sqrt(np.mean(np.power(imputed_X1 - imputed_X2,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class AllMissingValuesMultipleComponenetsMLEResultDoesntChangeTestCase(testing_utils.AllMissingBaseTestCase):
    """Tests that the maximum likelihood predictions, before and after fitting a three component GMM with MLE, are the same if all the data is missing.
    """
    def runTest(self):
        model = GMM(self.data, num_components=3, verbose=False, map_est=False)

        imputed_X1 = model.ml_imputation()

        model.fit()
        imputed_X2 = model.ml_imputation()

        rmse = np.sqrt(np.mean(np.power(imputed_X1 - imputed_X2,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):
    """Tests that the maximum likelihood predictions, before and after fitting a single component GMM with MLE to a single variable dataset,
     are the same if all the data is missing.
    """
    def runTest(self):

        model = GMM(self.data,  num_components=1, verbose=False, independent_vars=True, map_est=False)
        
        imputed_X1 = model.ml_imputation()

        model.fit()
        imputed_X2 = model.ml_imputation()

        rmse = np.sqrt(np.mean(np.power(imputed_X1 - imputed_X2,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneValueMLEResultTestCase(testing_utils.OneValueBaseTestCase):
    """Tests that the maximum-likelihood predictions of an MLE GMM are correct when there is only 1 non-missing value in each column.
    """
    def runTest(self):
        m0=np.zeros(shape=(self.data.shape[1], ))
        β0=1
        W0=np.eye(self.data.shape[1])*10000
        ν0=self.data.shape[1]
        model = GMM(self.data, num_components=1, verbose=False, map_est=False, m0=m0, ν0=ν0, β0=β0, W0=W0)

        model.fit(ϵ=0, max_iters=100)
        imputed_X = model.ml_imputation()
        
        rmse = np.sqrt(np.mean(np.power(imputed_X.flatten() - np.array([1, 2, 3]*3),2)))
        
        self.assertAlmostEqual(rmse, 0.0, places=6)

class TwoValueMLEResultTestCase(testing_utils.TwoValuesBaseTestCase):
    """Tests that the maximum-likelihood predictions of an MLE GMM are correct when there is only 2 non-missing value in each column.
    """
    def runTest(self):
        m0=np.zeros(shape=(self.data.shape[1], ))
        β0=1
        W0=np.eye(self.data.shape[1])*1000
        ν0=self.data.shape[1]
        model = GMM(self.data, num_components=1, verbose=False, map_est=False, m0=m0, ν0=ν0, β0=β0, W0=W0)

        model.fit(ϵ=0)
        imputed_X = model.ml_imputation()
        self.assertTrue(np.all(imputed_X == np.array([1, 3, 5, 6, 4, 2, 3.5, 3.5, 3.5]).reshape(3,3)))

class TwoValuesSamplesDifferentTestCase(testing_utils.TwoValuesBaseTestCase):
    """Tests that samples from the GMM are not identical.
    """
    def runTest(self):
        model = GMM(self.data, num_components=1, verbose=False, map_est=False)

        model.fit()
        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class IndependentVsDependentLLTestCase(testing_utils.IrisMCAR10BaseTestCase):
    """Tests that a GMM with a diagonal covariance matrices for the components performs worse than one with full covariance matrices,
     when the dataset has correlation between its variables.
    """
    def runTest(self):

        model_ind = GMM(self.data,  num_components=1, verbose=False, independent_vars=True, map_est=False)
        model_ind.fit()
        ll_ind = model_ind.log_likelihood(complete=False, return_mean=True)

        model_dep = GMM(self.data,  num_components=1, verbose=False, independent_vars=False, map_est=False)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertGreater(ll_dep, ll_ind)

class OneColumnPredTestCase(testing_utils.OneColumnBaseTestCase):
    """Tests that the GMM works for a dataset with a single column.
    """
    def runTest(self):
        m0=np.zeros(shape=(self.data.shape[1], ))
        β0=1
        W0=np.eye(self.data.shape[1])*1000
        ν0=self.data.shape[1]
        model = GMM(self.data, num_components=1, verbose=False, map_est=False, m0=m0, ν0=ν0, β0=β0, W0=W0)
        model.fit(ϵ=0)

        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([[1], [2], [1.5]])))

class OneColumnSampleTestCase(testing_utils.OneColumnBaseTestCase):
    """Tests that samples drawn for the same variable are different.
    """
    def runTest(self):

        model = GMM(self.data, num_components=1, verbose=False, map_est=False)
        model.fit()

        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class OneColumnLLTestCase(testing_utils.OneColumnBaseTestCase):
    """Tests that GMMS which have components with diagonal and full covariance matrices are equivalent on a single variable dataset.
    """
    def runTest(self):
        m0=np.zeros(shape=(self.data.shape[1], ))
        β0=1
        W0=np.eye(self.data.shape[1])*1000
        ν0=self.data.shape[1]
        model = GMM(self.data,  num_components=1, verbose=False, independent_vars=False, m0=m0, ν0=ν0, β0=β0, W0=W0)
        model.fit(ϵ=0)

        model_dep = GMM(self.data,  num_components=1, verbose=False, independent_vars=False, m0=m0, ν0=ν0, β0=β0, W0=W0)
        model_dep.fit(ϵ=0)
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertEqual(ll_dep, model.log_likelihood(complete=False, return_mean=True))

class TwoCompLLSmallerThan10CompTestCase(testing_utils.IrisMCAR10BaseTestCase):
    """Tests that a MLE GMM with 2 components has a lower missing data log-likelihood than one with 10 components.
    """
    def runTest(self):
        W0=np.eye(self.data.shape[1])
        model2 = GMM(self.data, num_components=2, verbose=False, independent_vars=True, map_est=False, W0=W0)
        model2.fit()
        ll2 = model2.log_likelihood(complete=False, return_mean=True)

        model10 = GMM(self.data, num_components=10, verbose=False, independent_vars=True, map_est=False, W0=W0)
        model10.fit()
        ll10 = model10.log_likelihood(complete=False, return_mean=True)
        
        self.assertGreater(ll10, ll2)

class MAPandMLEGiveDifferentLLsTestCase(testing_utils.IrisMCAR20BaseTestCase):
    """Tests that the MAP estimate and MLE for the GMM parameters are different.
    """
    def runTest(self):

        modelMLE = GMM(self.data,  num_components=2, verbose=False, independent_vars=True, map_est=False)
        modelMLE.fit()
        llMLE = modelMLE.log_likelihood(complete=False, return_mean=True)

        modelMAP = GMM(self.data,  num_components=2, verbose=False, independent_vars=True, map_est=True)
        modelMAP.fit()
        llMAP = modelMAP.log_likelihood(complete=False, return_mean=True)

        self.assertNotAlmostEqual(llMLE, llMAP)

class TenCompMAPDoesntCrashOnIris50TestCase(testing_utils.IrisMCAR50BaseTestCase):
    """Tests that a 10 component GMM with MAP estimation makes predictions for the Iris dataset with 50 MCAR data.
    """
    def runTest(self):

        model = GMM(self.data,  num_components=10, verbose=False, independent_vars=True, map_est=True)
        raised = False
        try:
            model.fit()
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')
