# James Allingham
# April 2018
# test_gmm.py
# Tests for the GMM module

import unittest
import sys
import numpy as np

# add both the relative and absolute paths for the code to test
sys.path.append("../auto_impute/")
sys.path.append("auto_impute/")

from gmm import GMM
import testing_utils

class BostonMCAR10LLTestCase(testing_utils.BostonMCAR10BaseTestCase):

    def runTest(self):
        model = GMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.28024941106327644)

class BostonMCAR30LLTestCase(testing_utils.BostonMCAR30BaseTestCase):

    def runTest(self):
        model = GMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.6490406849380511)
class IrisMCAR10LLTestCase(testing_utils.IrisMCAR10BaseTestCase):

    def runTest(self):
        model = GMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.16670302859010974)

class IrisMCAR30LLTestCase(testing_utils.IrisMCAR30BaseTestCase):

    def runTest(self):
        model = GMM(self.data, 3, verbose=False)
        model.fit()

        ll = model.log_likelihood()

        self.assertAlmostEqual(ll, -0.5453124374368254)
