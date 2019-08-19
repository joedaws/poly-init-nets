"""
misc.py

misc functions needed for polynomial intitialized networks
"""

import numpy as np

def check_const(idxset):
    """
    returns true if idx contains zero vector
    """
    flag = False
    for idx in idxset:
        if sum(np.abs(idx)) == 0:
            flag = True

    return flag
