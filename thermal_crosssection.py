"""
Functions to integrate the thermally averaged cross sections for BNV processes
"""

from constants import *

from scipy.special import kn
from scipy.integrate import quad
import numpy as np
from numpy import sqrt, power, pi
from collections.abc import Callable



class ThermalCrossSectionBNV:
    def __init__(self, mPsi: float,
                 mX: float,
                 mUp: float,
                 lambdak: float,
                 lambdaij: float):
        self.mPsi = mPsi
        self.mX = mX
        self.mUp = mUp
        self.lambdaLambdaPrime = lambdak * lambdaij
        self.lambdak = lambdak
        self.lambdaij = lambdaij
    
    # width
    def decay_width(self):
        return self.mX * (self.lambdaij**2 + self.lambdak**2) / (16 * pi)

    # Bare cross section for psi u -> d^c d^c
    def xs_psi_u_d_d(self, s):
        return self.lambdaLambdaPrime**2 / ((s - self.mX**2)**2 + (self.mX * self.decay_width())**2) \
            * (s - self.mPsi**2 - self.mUp**2) / (32*pi)

    # Bare cross section for psi d -> u^c d^c


    def thermal_xs_psi_u_d_d(self, T):
        def sigma(s):
            return self.xs_psi_u_d_d(s)
        
        return thermal_cross_section(T, sigma, ma=self.mPsi, mb=self.mUp)

    def thermal_xs_psi_d_u_d(self, T):
        pass




# Kallen helper function lambda(x, y, z) 
def kallen(x, y, z):
    return (x - y - z)**2 - 4*y*z


# Function to take the thermally averaged cross section for a process
# a + b -> X
# Takes in masses of a, b and passes in a cross section function of s
# Uses pCM^2 = kallen(s, ma^2, mb^2) / 4s
def thermal_cross_section(T: float, sigma: Callable, ma: float, mb: float) -> float:
    prefactor = 1 / (2 * T * power(ma * mb, 2) * kn(2, ma/T) * kn(2, mb/T))

    def integrand(s):
        return sqrt(s) * kn(1, sqrt(s)/T) * (kallen(s, ma**2, mb**2) / (4 * s)) * sigma(s)
    
    integral = quad(integrand, (ma + mb)**2, np.inf)[0]

    return prefactor * integral