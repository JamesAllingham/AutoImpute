# James Allingham
# May 2018
# test_cmm.py
# Tests for the CMM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from cmm import CMM
import testing_utils

class BostonMCAR10LLTestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):
        model = CMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -3.8682509264542033)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = CMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -8.377505572774568)

class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = CMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -1.9672464643080825)

class IrisMCAR30LLTestCase(testing_utils.IrisMCAR30BaseTestCase):

    def runTest(self):
        model = CMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, -3.4556377922939014)