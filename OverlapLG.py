# OverlapLG.py
# Function which implements the analytical overlap of two Laguerre Gauss modes. 

import time
import numpy as np 
import scipy as sp
from scipy import special
from scipy import integrate
import math
from math import factorial, sqrt
import matplotlib.pyplot as plt 

# Define constants

cm = 1e-2
mm = 1e-3
nm = 1e-9
lambda0 = 810*nm 
w0 = 1*mm
k0 = 2*np.pi/lambda0
zR0 = (np.pi*w0**2)/lambda0
epi = 1e-12

def overlapIntegral(l,p,p2,z,z2,w,w2):

    wz = lambda z,w0: w0*sqrt(1+(z/(np.pi*w0**2/lambda0))**2)
    zz = lambda z,w0: z/(np.pi*w0**2/lambda0)

    # Functions which encodes LG modes

    normCoeff = lambda ell, p: ((((-1)**p)*sqrt(2)*sqrt(factorial(p)))/(sqrt(np.pi)*sqrt(factorial(p+abs(ell)))))*2**(abs(ell)/2)
    
    # Radial part of LG mode

    radLGExpPart = lambda rho,z,w0: -(rho/wz(z,w0))**2 + (1j*k0*rho**2)/(2*z*(1+(1/zz(z,w0))**2))
    radLGExpPartConj = lambda rho,z,w0: -(rho/wz(z,w0))**2 - (1j*k0*rho**2)/(2*z*(1+(1/zz(z,w0))**2))
    radLG = lambda rho, ell, p, z, w0: normCoeff(ell,p)*special.genlaguerre(p,abs(ell))(2*rho**2/(w0**2*(1+zz(z,w0)**2)))*np.exp(radLGExpPart(rho,z,w0))*(1/wz(z,w0))**(1+abs(ell))*rho**abs(ell)
    radLGConj = lambda rho, ell, p, z, w0: normCoeff(ell,p)*special.genlaguerre(p,abs(ell))(2*rho**2/(w0**2*(1+zz(z,w0)**2)))*np.exp(radLGExpPartConj(rho,z,w0))*(1/wz(z,w0))**(1+abs(ell))*rho**abs(ell)
    
    # Parts of the analytical integration expression 

    # aCoeff = lambda l,p,l2,p2,z,z2,w,w2: (np.conjugate(normCoeff(l,p))*normCoeff(l2,p2)*(1/wz(z,w))**(1+abs(l))*(1/wz(z2,w2))**(1+abs(l2))) 
    
    aCoeff = lambda l,p,l2,p2,z,z2,w,w2: (np.conjugate(normCoeff(l,p))*normCoeff(l2,p2)*(1/wz(z,w))**(1+abs(l))*(1/wz(z2,w2))**(1+abs(l2)))
    sigma = lambda z,z2,w,w2: 1/wz(z,w)**2 + 1/wz(z2,w2)**2 + 1j*(k0/2)*(-1/(z*(1+(1/zz(z,w))**2)) + 1/(z2*(1+(1/zz(z2,w2))**2)))
    lambdaZ = lambda z,w: 2/(w**2*(1+zz(z,w)**2)) 
    mu = lambda l,l2: (abs(l)+abs(l2))/2
    
    overlapIntegral = lambda l,p,p2,z,z2,w,w2: aCoeff(l,p,l,p2,z,z2,w,w2)*(special.gamma(p+p2+abs(l)+1)/(factorial(p)*factorial(p2)))*((sigma(z,z2,w,w2) - lambdaZ(z2,w2))**p2*(sigma(z,z2,w,w2) - lambdaZ(z,w))**p)/((sigma(z,z2,w,w2))**(p+p2+abs(l)+1))*special.hyp2f1(-p,-p2,-p-p2-abs(l),(sigma(z,z2,w,w2)*(sigma(z,z2,w,w2)-lambdaZ(z,w)-lambdaZ(z,w2))/((sigma(z,z2,w,w2)-lambdaZ(z2,w2))*(sigma(z,z2,w,w2)-lambdaZ(z,w)))))
    
    return overlapIntegral(l,p,p2,z,z2,w,w2)

# example call 

# print(overlapIntegral(2, 0, 1, 0.5, 0.6, 1*mm, 1.1*mm))

