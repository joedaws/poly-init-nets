"""
roots.py

Tools to generate lists of roots of several polynomials.
Lists of roots are stored in the data directory
"""

import numpy as np
import csv 
import numpy.polynomial.legendre as leg
from polydata import polydata

# set max degree
MAXD = polydata['MAXDEGREE']
LEGF = polydata['LEGFILE']

# function to genereate Legendre 
def get_leg_roots():
    roots = {}
    # loop over all degrees up to MAXD
    for i in range(0,MAXD):
        c = np.zeros(MAXD)
        c[i] = 1
        r = leg.legroots(c)
        roots.update({i:r})
   
    # save roots to file
    w = csv.writer(open(LEGF,"w"))
    for key, val in roots.items():
        w.writerow([key,val])

    return 
