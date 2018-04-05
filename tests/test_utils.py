# James Allingham
# April 2018
# test_utils.py
# Common utils for the unit tests.

import unittest
import numpy as np

class NoMissingValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        # create a fake data file to impute
        self.data = np.array([[   1,-2.5,  0],
                              [  12,   5,6.5],
                              [-7.5,  10, -9],
                              [   0,   0,  4]],
                              dtype=np.float32)

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
             [6     , 4     , 2     ],
             [np.nan, np.nan, np.nan]],
             dtype=np.float32
        )

# Temporary test cases - not permanent features, just here to make sure I don't break things while refactoring code
class BostonMCAR10BaseTestCase(unittest.TestCase):

    def setUp(self):        
        try:
            self.data = np.genfromtxt("../data/boston-10-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/boston-10-MCAR.csv", delimiter=",")

class BostonMCAR20BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/boston-20-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/boston-20-MCAR.csv", delimiter=",")

class BostonMCAR30BaseTestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/boston-30-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/boston-30-MCAR.csv", delimiter=",")

class BostonMCAR40BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/boston-40-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/boston-40-MCAR.csv", delimiter=",")

class BostonMCAR50BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/boston-50-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/boston-50-MCAR.csv", delimiter=",")

class IrisMCAR10BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/iris-10-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/iris-10-MCAR.csv", delimiter=",")

class IrisMCAR20BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/iris-20-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/iris-20-MCAR.csv", delimiter=",")

class IrisMCAR30BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/iris-30-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/iris-30-MCAR.csv", delimiter=",")

class IrisMCAR40BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/iris-40-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/iris-40-MCAR.csv", delimiter=",")

class IrisMCAR50BaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.data = np.genfromtxt("../data/iris-50-MCAR.csv", delimiter=",")
        except FileNotFoundError as e:
            self.data = np.genfromtxt("data/iris-50-MCAR.csv", delimiter=",")
