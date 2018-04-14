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

def get_locs_and_coords(mask_row):

    o_locs = np.where(~mask_row)[0]
    m_locs = np.where(mask_row)[0]
    oo_coords = tuple(zip(*[(i, j) for i in o_locs for j in o_locs]))
    mm_coords = tuple(zip(*[(i, j) for i in m_locs for j in m_locs]))
    mo_coords = tuple(zip(*[(i, j) for i in m_locs for j in o_locs]))
    om_coords = tuple(zip(*[(i, j) for i in o_locs for j in m_locs]))

    return o_locs, m_locs, oo_coords, mm_coords, mo_coords, om_coords