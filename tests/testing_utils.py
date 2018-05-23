# James Allingham
# April 2018
# test_utils.py
# Common utils for the unit tests.

import unittest
import numpy as np
import numpy.ma as ma

class NoRowsBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.zeros(shape=(0,3), dtype=np.float32)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class NoColsBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.zeros(shape=(3,0), dtype=np.float32)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class NoMissingValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.array([[   1,-2.5,  0],
                         [  12,   5,6.5],
                         [-7.5,  10, -9],
                         [   0,   0,  4]],
                              dtype=np.float32)
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class AllMissingBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan]],
             dtype=np.float32
        )
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)
        
class OneValueBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[1     , np.nan, np.nan],
             [np.nan, 2     , np.nan],
             [np.nan, np.nan, 3     ]],
             dtype=np.float32
        )
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class TwoValuesBaseTestCase(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[1     , 3     , 5     ],
             [6     , 4     , 2     ],
             [np.nan, np.nan, np.nan]],
             dtype=np.float32
        )
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

# Temporary test cases - not permanent features, just here to make sure I don't break things while refactoring code
class BostonMCAR10BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/boston-10-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/boston-10-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class BostonMCAR20BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/boston-20-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/boston-20-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class BostonMCAR30BaseTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/boston-30-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/boston-30-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class BostonMCAR40BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/boston-40-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/boston-40-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class BostonMCAR50BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/boston-50-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/boston-50-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class IrisMCAR10BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/iris-10-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/iris-10-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class IrisMCAR20BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/iris-20-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/iris-20-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class IrisMCAR30BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/iris-30-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/iris-30-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class IrisMCAR40BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/iris-40-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/iris-40-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)

class IrisMCAR50BaseTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        try:
            data = np.genfromtxt("../data/iris-50-MCAR.csv", delimiter=",")
        except FileNotFoundError as _:
            data = np.genfromtxt("data/iris-50-MCAR.csv", delimiter=",")
        mask = np.isnan(data)
        self.data = ma.masked_array(data, mask)
