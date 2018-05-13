# James Allingham
# May 2018
# test_bgmm.py
# Tests for the BGMM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from bgmm import BGMM
import testing_utils

class BostonMCAR10LLTestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):
        model = BGMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, 2.4515782517309197)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = BGMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, 5.43323273778184)
class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = BGMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, 0.9511341230319695)

class IrisMCAR30LLTestCase(testing_utils.IrisMCAR30BaseTestCase):

    def runTest(self):
        model = BGMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertGreaterEqual(ll, 1.4069742295142373)
