# James Allingham
# March 2018
# utilities.py
# Common code shared between the various models.

from sys import stderr
import numpy as np
from scipy import linalg

def regularise_Σ(Σ):
    it = 0
    while not np.all(np.linalg.eigvals(Σ) >= 0) or linalg.det(Σ) == 0:
        Σ += np.eye(Σ.shape[0])*10**(-3 + it)
        it += 1

    return Σ

def print_err(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def encode_1_hot(i, n):
    tmp = np.zeros((n,))
    tmp[i] = 1
    return tmp