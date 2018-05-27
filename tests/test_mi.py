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

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesLLTestCase(testing_utils.NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)  

        lls = model.log_likelihood(complete=True, return_individual=True)

        self.assertTrue(np.all(lls < 0))

class OneValueResultTestCase(testing_utils.OneValueBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.array_equal(result, np.array([1,2,3,1,2,3,1,2,3]).reshape(3,3)))

class TwoValuesResultTestCase(testing_utils.TwoValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.array_equal(result, np.array([1,3,5,6,4,2,3.5,3.5,3.5]).reshape(3,3)))

class AllMissingValuesTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.ml_imputation()

        self.assertTrue(np.all(result == 0))

class AllMissingValuesLLTestCase(testing_utils.AllMissingBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)  

        lls = model.log_likelihood(complete=True, return_individual=True)

        self.assertTrue(np.all(lls > 0))

class NoRowsTestCase(testing_utils.NoRowsBaseTestCase):

    def runTest(self):
        with self.assertRaises(SystemExit):
            MeanImpute(self.data, verbose=False)

class NoColsTestCase(testing_utils.NoColsBaseTestCase):

    def runTest(self):
        with self.assertRaises(SystemExit):
            MeanImpute(self.data, verbose=False)

class OneColumnAllMissingTestCase(testing_utils.OneColumnAllMissingBaseTestCase):

    def runTest(self):

        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.ml_imputation()
        rmse = np.sqrt(np.mean(np.power(np.zeros(shape=(3,1)) - imputed_X,2)))
        
        self.assertAlmostEqual(rmse, 0.0)