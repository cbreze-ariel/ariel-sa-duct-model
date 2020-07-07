import math
import numpy as np

'''
Antiporter Fxn 
ap(ao,ai,bo,bi,ka,kb) in original
'''
def antiporter(ao,ai,bo,bi,ka,kb):
    numerator = ao*bi-bo*ai
    denominator = (ka*kb*((1+ai/ka+bi/kb)*(ao/ka+bo/kb)+\
                          (1+ao/ka+bo/kb)*(ai/ka+bi/kb)))
    return (numerator/denominator)

'''
Effective Permeability Fxn (g(xi,xo) in original)
Linearization of the Constant Field Eqn
'''
def eff_perm(xi,xo):
    if xi > xo:
        print("numerator", xi, "denominator", xo)
    return (xi*xo*np.log(xi/xo)/(xi-xo)) # Natural Log

'''
Nernst Potential Fxn
'''
def nernst(concentrationA, concentrationB):
    # Physical Constants
    ideal_gas = 8.31451 # J mol^-1 K^-1
    faraday_cst = 96485 # C mol^-1
    body_temp = 310 #K
    return (ideal_gas*body_temp/faraday_cst)*np.log(concentrationA/concentrationB) # Natural Log