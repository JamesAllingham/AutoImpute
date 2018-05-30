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
        model = SingleGaussian(self.data, verbose=False, map_est=False)
        model.fit()

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

# class NoMissingValuesCompleteInitialLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

#     def runTest(self):
#         model = SingleGaussian(self.data, verbose=False, independent_vars=True, map_est=False)

#         ll = model.log_likelihood(return_mean=True, complete=True)

#         self.assertAlmostEqual(ll, -3.179506983107214)

# class NoMissingValuesCompleteFinalLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

#     def runTest(self):
#         model = SingleGaussian(self.data, verbose=False, map_est=False)

#         ll1 = model.log_likelihood(return_mean=True, complete=True)
#         model.fit()
#         ll2 = model.log_likelihood(return_mean=True, complete=True)
#         print(ll1, ll2)
#         self.assertAlmostEqual(ll1, ll2)

class NoMissingValuesMissingLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, map_est=False)

        ll = model.log_likelihood(return_mean=True, complete=False)

        self.assertTrue(np.isnan(ll))

class AllMissingValuesMLEResultDoesntChangeTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, map_est=False)

        imputed_X1 = model.ml_imputation()

        model.fit()
        imputed_X2 = model.ml_imputation()

        rmse = np.sqrt(np.mean(np.power(imputed_X1 - imputed_X2,2)))
        
        self.assertAlmostEqual(rmse, 0.0)

class OneValueMLEResultTestCase(testing_utils.OneValueBaseTestCase):

    def runTest(self):
        D = self.data.shape[1]
        W0=np.eye(D)*10000
        m0=np.zeros(shape=(D,))
        model = SingleGaussian(self.data, verbose=False, map_est=False, W0=W0, m0=m0)

        model.fit(ϵ=0, max_iters=100)
        imputed_X = model.ml_imputation()

        rmse = np.sqrt(np.mean(np.power(imputed_X.flatten() - np.array([1, 2, 3]*3),2)))
        
        self.assertAlmostEqual(rmse, 0.0, places=6)

class TwoValueMLEResultTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        D = self.data.shape[1]
        W0=np.eye(D)*10000
        m0=np.zeros(shape=(D,))
        model = SingleGaussian(self.data, verbose=False, map_est=False, W0=W0, m0=m0)

        model.fit(ϵ=0, max_iters=100)
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 3, 5, 6, 4, 2, 3.5, 3.5, 3.5]).reshape(3,3)))

class TwoValueMAPResultTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        D = self.data.shape[1]
        m0=np.zeros(shape=(D,))
        β0=1
        W0=np.eye(D)*10000
        ν0=D
        model = SingleGaussian(self.data, verbose=False, map_est=True, m0=m0, β0=β0, ν0=ν0, W0=W0)
        model.fit(ϵ=0, max_iters=100)
        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([1, 3, 5, 6, 4, 2, 2, 2, 2]).reshape(3,3)))

class TwoValuesSamplesDifferentTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = SingleGaussian(self.data, verbose=False, map_est=False)

        model.fit()
        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class IndependentVsDependentLLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        D = self.data.shape[1]
        W0=np.eye(D)*10000
        m0=np.zeros(shape=(D,))
        model_ind = SingleGaussian(self.data, verbose=False, independent_vars=True, map_est=False, W0=W0, m0=m0)
        model_ind.fit()
        ll_ind = model_ind.log_likelihood(complete=False, return_mean=True)

        model_dep = SingleGaussian(self.data, verbose=False, independent_vars=False, map_est=False, W0=W0, m0=m0)
        model_dep.fit()
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertGreater(ll_dep, ll_ind)

class OneColumnPredTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False, map_est=False)
        model.fit(ϵ=0, max_iters=100)

        imputed_X = model.ml_imputation()

        self.assertTrue(np.all(imputed_X == np.array([[1], [2], [1.5]])))

class OneColumnSampleTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False, map_est=False)
        model.fit()

        samples = model.sample(2)
        rmse = np.sqrt(np.mean(np.power(samples[0, :, :] - samples[1, :, :],2)))
        self.assertTrue(rmse > 0)

class OneColumnLLTestCase(testing_utils.OneColumnBaseTestCase):

    def runTest(self):
        m0=np.zeros(shape=(self.data.shape[1], ))
        β0=1
        W0=np.eye(self.data.shape[1])*1000
        ν0=self.data.shape[1]
        model = SingleGaussian(self.data, verbose=False, independent_vars=False, map_est=False, m0=m0, ν0=ν0, β0=β0, W0=W0)
        model.fit(ϵ=0)

        model_dep = SingleGaussian(self.data, verbose=False, independent_vars=True, map_est=False, m0=m0, ν0=ν0, β0=β0, W0=W0)
        model_dep.fit(ϵ=0)
        ll_dep = model_dep.log_likelihood(complete=False, return_mean=True)

        self.assertEqual(ll_dep, model.log_likelihood(complete=False, return_mean=True))

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):

    def runTest(self):

        model = SingleGaussian(self.data, verbose=False, independent_vars=False, map_est=False)
        imputed_X1 = model.ml_imputation()
        model.fit()
        imputed_X2 = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(imputed_X1 - imputed_X2,2)))
        
        self.assertAlmostEqual(rmse, 0.0)
