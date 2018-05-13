# James Allingham
# May 2018
# test_cmm.py
# Tests for the MMM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from mmm import MMM
import testing_utils

class BostonMCAR10LLTestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):
        model = MMM(self.data, 3, verbose=False, assignments="")
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.3898741790069437)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = MMM(self.data, 3, verbose=False, assignments="")
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -1.4240416759355332)

class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = MMM(self.data, 3, verbose=False, assignments="")
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.3805897888979955)

class IrisMCAR30LLTestCase(testing_utils.IrisMCAR30BaseTestCase):

    def runTest(self):
        model = MMM(self.data, 3, verbose=False, assignments="")
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -0.49930916896351535)
