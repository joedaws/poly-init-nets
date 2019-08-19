#!/usr/bin/python3
"""
roots.py

Tools to generate lists of roots of several polynomials.
Lists of roots are stored in the data directory
"""

import numpy as np
import csv 
import numpy.polynomial.legendre as leg
import numpy.polynomial.chebyshev as cheb

# set this dictionary of useful terms
polydata = {'MAXDEGREE': 20,
            'LEGFILE':"../data/polyroots/legroots.txt",
            'CHEBFILE':"../data/polyroots/chebroots.txt"}

# set some global variables
MAXD = polydata['MAXDEGREE']
LEGF = polydata['LEGFILE']
CHEBF = polydata['CHEBFILE']

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

# function to genereate Legendre 
def leg_roots_vec():
    roots = {}
    # loop over all degrees up to MAXD
    for i in range(0,MAXD):
        c = np.zeros(MAXD)
        c[i] = 1
        r = leg.legroots(c)
        roots.update({i:r})
   
    return roots

# function to get scaling factor in legendre polynomial
def get_leg_scalfac(deg):
    testpt = 0.5
    c = np.zeros(MAXD)
    c[deg] = 1
    val = leg.legval(testpt,c,tensor=False)
    r = leg.legroots(c)
    prod = 1
    for root in r:
        prod = prod * (testpt-root)

    scalfac = val/prod

    return scalfac

# function to genereate Chebyshev Roots 
def get_cheb_roots():
    roots = {}
    # loop over all degrees up to MAXD
    for i in range(0,MAXD):
        c = np.zeros(MAXD)
        c[i] = 1
        r = cheb.chebroots(c)
        roots.update({i:r})
   
    # save roots to file
    w = csv.writer(open(CHEBF,"w"))
    for key, val in roots.items():
        w.writerow([key,val])

    return 

if (__name__ == '__main__'):
    get_leg_roots()
    get_cheb_roots()
