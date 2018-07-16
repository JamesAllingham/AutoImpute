# James Allingham
# April 2018
# test_mi.py
# Tests for the mi module

import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from mi import MeanImpute
import testing_utils

class NoMissingValuesRMSETestCase(testing_utils.NoMissingValuesBaseTestCase):
    """Tests that when there are no missing values the prediction RMSE is 0.
    """

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesLLTestCase(testing_utils.NoMissingValuesBaseTestCase):
    """Tests that the missing data log-likelihoods for a toy dataset are correct.
    """
    def runTest(self):
        model = MeanImpute(self.data, verbose=False)  

        lls = model.log_likelihood(complete=True, return_individual=True)

        self.assertTrue(np.all(lls < 0))

class OneValueResultTestCase(testing_utils.OneValueBaseTestCase):
    """Tests that when there is only one non-missing value, the mean imputation will return exactly that value.
    """
    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.array_equal(result, np.array([1,2,3,1,2,3,1,2,3]).reshape(3,3)))

class TwoValuesResultTestCase(testing_utils.TwoValuesBaseTestCase):
    """Tests that when there are only two non-missing values, the mean imputation will return the correct value.
    """
    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.array_equal(result, np.array([1,3,5,6,4,2,3.5,3.5,3.5]).reshape(3,3)))

class AllMissingValuesTestCase(testing_utils.AllMissingBaseTestCase):
    """Tests that when there are no observed values the results of mean imputation are all 0s.
    """
    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.all(result == 0))

class AllMissingValuesLLTestCase(testing_utils.AllMissingBaseTestCase):
    """Tests that when there are no observed values the missing data log-likelihoods are correct.
    """
    def runTest(self):
        model = MeanImpute(self.data, verbose=False)  

        lls = model.log_likelihood(complete=True, return_individual=True)

        self.assertTrue(np.all(lls > 0))

class NoRowsTestCase(testing_utils.NoRowsBaseTestCase):
    """Tests that if a dataset with no rows is the input then an error will be thrown.
    """
    def runTest(self):
        with self.assertRaises(SystemExit):
            MeanImpute(self.data, verbose=False)

class NoColsTestCase(testing_utils.NoColsBaseTestCase):
    """Tests that if a dataset with no columns is the input then an error will be thrown.
    """
    def runTest(self):
        with self.assertRaises(SystemExit):
            MeanImpute(self.data, verbose=False)

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):
    """Tests that the mean imputation works if given only 1 column.
    """
    def runTest(self):

        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,1)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)