# James Allingham
# April 2018
# test_utils.py
# Common utils for the unit tests.

import unittest
import numpy as np

class NoMissingValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        # create a fake data file to impute
        self.data = np.array(
            [[1,2,3],
             [4,5,6],
             [7,8,9]], 
             dtype=np.float32
        )

class OneValueBaseTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.array(
            [[1     , np.nan, np.nan],
             [np.nan, 2     , np.nan],
             [np.nan, np.nan, 3     ]],
             dtype=np.float32
        )

class TwoValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        self.data = np.array(
            [[1     , 3     , 5     ],
             [2     , 4     , 6     ],
             [np.nan, np.nan, np.nan]],
             dtype=np.float32
        )
        