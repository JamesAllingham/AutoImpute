# James Allingham
# April 2018
# test_mi.py
# Tests for the mi module

import unittest
import sys
import numpy as np
import numpy.ma as ma 

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from mi import MeanImpute
from testing_utils import NoMissingValuesBaseTestCase, OneValueBaseTestCase,TwoValuesBaseTestCase

class NoMissingValuesRMSETestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.impute()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertGreaterEqual(rmse, 0.0)

class NoMissingValuesLLTestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertTrue(np.isnan(ll))

class OneValueLLTestCase(OneValueBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, 7.604817318859187)

class OneValueResultTestCase(OneValueBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.impute()

        self.assertTrue(np.array_equal(result, np.array([1,2,3,1,2,3,1,2,3]).reshape(3,3)))

class TwoValuesLLTestCase(TwoValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, 7.604817318859187)

class TwoValuesResultTestCase(TwoValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.impute()

        self.assertTrue(np.array_equal(result, np.array([1,3,5,6,4,2,3.5,3.5,3.5]).reshape(3,3)))

class AllMissingValuesTestCase(unittest.TestCase):

    def setUp(self):
        data = np.array([np.nan]*9).reshape(3,3)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=False)

class NoRowsTestCase(unittest.TestCase):

    def setUp(self):
        data = np.zeros(shape=(0,3), dtype=np.float32)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=True)

class NoColsTestCase(unittest.TestCase):

    def setUp(self):
        data = np.zeros(shape=(3,0), dtype=np.float32)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=True)