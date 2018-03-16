# James Allingham
# Mar 2018
# data_gen.py
# This script generates the test data sets, with 10-50% (in increments of 10) missing entries:
# 1. Boston Housing
# 2. Iris

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.datasets import load_boston, load_iris

def main():
    datasets = [("boston", load_boston), ("iris", load_iris)]
    
    for name, load_fun in datasets:
        X,_ = load_fun(return_X_y=True)
        for i in range(0, 51, 10):
            mask = np.random.rand(*X.shape) <= i/100
            masked_X = X.copy()
            masked_X[mask] = np.NaN
            np.savetxt("%s-%s-MCAR.csv" % (name, i), masked_X, delimiter=",")

if __name__ == "__main__":
    main()
