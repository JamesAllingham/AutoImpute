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