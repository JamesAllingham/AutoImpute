# James Allingham
# April 2018
# test_MeanImpute.py
# Tests for the mean imputation module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from MeanImpute import MeanImpute
from test_utils import NoMissingValuesBaseTestCase

class NoMissingValuesRMSETestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        imputed_X = model.impute()
        rmse = np.sqrt(np.mean(np.power(self.data - imputed_X,2)))

        self.assertAlmostEqual(rmse, 0.0)

class NoMissingValuesLLTestCase(NoMissingValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, -np.inf)

class OneValueBaseTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.array(
            [[1     , np.nan, np.nan],
             [np.nan, 2     , np.nan],
             [np.nan, np.nan, 3     ]],
             dtype=np.float32
        )

class OneValueLLTestCase(OneValueBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, 0)

class OneValueResultTestCase(OneValueBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.impute()

        self.assertTrue(np.array_equal(result, np.array([1,2,3,1,2,3,1,2,3]).reshape(3,3)))

class TwoValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.array(
            [[1     , 3     , 5     ],
             [2     , 4     , 6     ],
             [np.nan, np.nan, np.nan]],
             dtype=np.float32
        )

class TwoValuesLLTestCase(TwoValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        ll = model.log_likelihood()

        self.assertEqual(ll, -np.inf)

class TwoValuesResultTestCase(TwoValuesBaseTestCase):

    def runTest(self):
        model = MeanImpute(self.data, verbose=False)

        result = model.impute()

        self.assertTrue(np.array_equal(result, np.array([1,3,5,2,4,6,1.5,3.5,5.5]).reshape(3,3)))

class AllMissingValuesTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.array([np.nan]*9).reshape(3,3)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=False)

class NoRowsTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.zeros(shape=(0,3), dtype=np.float32)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=True)

class NoColsTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.zeros(shape=(3,0), dtype=np.float32)

    def runTest(self):
        with self.assertRaises(RuntimeError):
            MeanImpute(self.data, verbose=True)