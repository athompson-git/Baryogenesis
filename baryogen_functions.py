import numpy as np
from scipy.special import zeta
from numpy import power, pi, sqrt

from constants import *


# number of dof
gstar_dat = np.genfromtxt("data/Gstar1.dat")
def gstar(T):
    return np.interp(T, gstar_dat[:,0], gstar_dat[:,1])




# Solve for the asymmetry parameter epsilon for X species alpha
def epsilon_alpha(lamij_1, lamij_2, lamk_1, lamk_2, mX1, mX2):
    # lamij, lamk are complex
    return ((lamij_1.imag * lamk_1.imag * lamk_2 * lamij_2) / (abs(lamk_1)**2 + abs(lamij_1)**2) / (8*np.pi)) \
        * ((mX1**2 - mX2**2)*mX1*mX2 / (np.power(mX1**2 - mX2**2, 2) + np.power(mX1 * abs(lamij_2)**2 * mX2 / 12 / np.pi, 2) ))




# Functions for nnbar oscillation
LamQCD = 0.15
HBAR = 6.58212e-25

def Gnn(lam1, lam13, mPsi, mX):
    return mPsi * power(lam1*lam13**2, 2) * np.log(power(mX/mPsi, 2)) / (16*pi**2) / power(mX, 6)

def tauNNBar(lam1, lam13, mPsi, mX):
    return HBAR*power(power(LamQCD, 6)*Gnn(lam1, lam13, mPsi, mX), -1)



# Simplified asymmetry parameters for two X species with couplings lam and lamPrime
def eps1_lam_lamPrime(lam, lamPrime, alpha, theta=np.pi/2):
    lam_lamPrime_sum = lam**2 + lamPrime**2

    prefactor = 1/(8 * pi * lam_lamPrime_sum)
    return abs(prefactor * power(lam*lamPrime, 2) * np.sin(theta) * (1 - alpha**2)*alpha \
        / (power(1-alpha**2, 2) + power(alpha, 2)*power(lam_lamPrime_sum, 2)/power(16*pi, 2)))

def eps2_lam_lamPrime(lam, lamPrime, alpha, theta=np.pi/2):
    lam_lamPrime_sum = lam**2 + lamPrime**2

    prefactor = 1/(8 * pi * lam_lamPrime_sum)
    return abs(prefactor * power(lam*lamPrime, 2) * np.sin(theta) * (alpha**2 - 1)*alpha \
        / (power(1-alpha**2, 2) + power(lam_lamPrime_sum, 2)/power(16*pi, 2)))




# Washout constraint
def lam_washout(lam, T, mX):
    # returns lamij/mx^2 limit in TeV^-2
    # takes T in GeV
    return mX**2 * 1e6 * sqrt(864 * power(pi, 7/2) * zeta(3) / sqrt(5) / M_PL / T**3) / lam
