import jax
jax.config.update('jax_enable_x64', True)

# OverlapLG_jax.py
# Function which implements the analytical overlap of two Laguerre Gauss modes, JAX version.

import time
import jax.numpy as jnp
import scipy as sp
#from scipy import special
from scipy import integrate
import math
#from math import sqrt
import matplotlib.pyplot as plt 
import time
from hyp2f1_jax import hyp2f1
from jax.scipy.special import gammaln, gamma

import jax
import jax.numpy as jnp

@jax.jit
def gen_laguerre(n, alpha, x):
    """
    Computes the generalized Laguerre polynomial L_n^{(alpha)}(x)
    using recurrence, supports scalar or array x
    """
    def body(k, vals):
        L_nm2, L_nm1 = vals
        L_n = ((2*k - 1 + alpha - x) * L_nm1 - (k - 1 + alpha) * L_nm2) / k
        return (L_nm1, L_n)

    L0 = jnp.ones_like(x)
    if n == 0:
        return L0

    L1 = 1 + alpha - x
    if n == 1:
        return L1

    _, L_n = jax.lax.fori_loop(2, n + 1, body, (L0, L1))
    return L_n


# JIT-friendly factorial
@jax.jit
def factorial(n):
    return jnp.exp(gammaln(n + 1))

# Define constants

cm = 1e-2
mm = 1e-3
nm = 1e-9
lambda0 = 810*nm 
w0 = 1*mm
k0 = 2*jnp.pi/lambda0
zR0 = (jnp.pi*w0**2)/lambda0
epi = 1e-12

@jax.jit
def overlapIntegral(l,p,p2,z,z2,w,w2):

    wz = lambda z,w0: w0*jnp.sqrt(1+(z/(jnp.pi*w0**2/lambda0))**2)
    zz = lambda z,w0: z/(jnp.pi*w0**2/lambda0)

    # Functions which encodes LG modes
    normCoeff = lambda ell, p: ((((-1)**p)*jnp.sqrt(2)*jnp.sqrt(factorial(p)))/(jnp.sqrt(jnp.pi)*jnp.sqrt(factorial(p+abs(ell)))))*2**(abs(ell)/2)
    
    # Radial part of LG mode

    radLGExpPart = lambda rho,z,w0: -(rho/wz(z,w0))**2 + (1j*k0*rho**2)/(2*z*(1+(1/zz(z,w0))**2))
    radLGExpPartConj = lambda rho,z,w0: -(rho/wz(z,w0))**2 - (1j*k0*rho**2)/(2*z*(1+(1/zz(z,w0))**2))
    radLG = lambda rho, ell, p, z, w0: normCoeff(ell,p)*gen_laguerre(p,abs(ell))(2*rho**2/(w0**2*(1+zz(z,w0)**2)))*jnp.exp(radLGExpPart(rho,z,w0))*(1/wz(z,w0))**(1+abs(ell))*rho**abs(ell)
    radLGConj = lambda rho, ell, p, z, w0: normCoeff(ell,p)*gen_laguerre(p,abs(ell))(2*rho**2/(w0**2*(1+zz(z,w0)**2)))*jnp.exp(radLGExpPartConj(rho,z,w0))*(1/wz(z,w0))**(1+abs(ell))*rho**abs(ell)
    
    # Parts of the analytical integration expression 

    aCoeff = lambda l,p,l2,p2,z,z2,w,w2: (jnp.conjugate(normCoeff(l,p))*normCoeff(l2,p2)*(1/wz(z,w))**(1+abs(l))*(1/wz(z2,w2))**(1+abs(l2)))
    sigma = lambda z,z2,w,w2: 1/wz(z,w)**2 + 1/wz(z2,w2)**2 + 1j*(k0/2)*(-1/(z*(1+(1/zz(z,w))**2)) + 1/(z2*(1+(1/zz(z2,w2))**2)))
    lambdaZ = lambda z,w: 2/(w**2*(1+zz(z,w)**2)) 
    mu = lambda l,l2: (abs(l)+abs(l2))/2
    
    overlapIntegral = lambda l,p,p2,z,z2,w,w2: aCoeff(l,p,l,p2,z,z2,w,w2)*(gamma(p+p2+abs(l)+1)/(factorial(p)*factorial(p2)))*((sigma(z,z2,w,w2) - lambdaZ(z2,w2))**p2*(sigma(z,z2,w,w2) - lambdaZ(z,w))**p)/((sigma(z,z2,w,w2))**(p+p2+abs(l)+1))*hyp2f1(-p,-p2,-p-p2-abs(l),(sigma(z,z2,w,w2)*(sigma(z,z2,w,w2)-lambdaZ(z,w)-lambdaZ(z,w2))/((sigma(z,z2,w,w2)-lambdaZ(z2,w2))*(sigma(z,z2,w,w2)-lambdaZ(z,w)))))
    
    return overlapIntegral(l,p,p2,z,z2,w,w2)

# example call 


_ = overlapIntegral(2, 0, 1, 0.5, 0.6, 1*mm, 1.1*mm)

start_time = time.time()
print(overlapIntegral(2, 0, 1, 0.5, 0.6, 1*mm, 1.1*mm))
print(time.time() - start_time)