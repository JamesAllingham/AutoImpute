# James Allingham
# March 2018
# utilities.py
# Common code shared between the various models.
import numpy as np

def regularise_Σ(Σ):
    it = 0
    while not np.all(np.linalg.eigvals(Σ) >= 0):
        Σ += np.eye(Σ.shape[0])*10**(-3 + it)
        it += 1

    return Σ