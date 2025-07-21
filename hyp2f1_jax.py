import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gamma
from scipy import special
import time

'''
Implementation of the Gaussian Hypergeometric Function with JAX.  
'''

# Safer Pochhammer (a)_n using gamma function

@jax.jit
def pochhammer(a, n):
    return jnp.exp(gammaln(a + n) - gammaln(a))

@jax.jit
def pochhammer_safe(a, n):
    def body_fun(ii, result):
        return result * (a + ii)
    return jax.lax.fori_loop(0, n, body_fun, 1.0)

# Taylor series for |z| < 1
@jax.jit
def hyp2f1_taylor(a, b, c, z, max_terms=100, tol=1e-12):
    def term(n):
        num = pochhammer(a, n) * pochhammer(b, n)
        denom = pochhammer(c, n) * jnp.exp(gammaln(n + 1))
        return (num / denom) * z**n

    def cond(state):
        n, acc, t = state
        return jnp.logical_and(n < max_terms, jnp.abs(t) > tol)

    def body(state):
        n, acc, _ = state
        t = term(n)
        return n + 1, acc + t, t

    init = (0, jnp.zeros_like(z, dtype=jnp.complex64), jnp.ones_like(z, dtype=jnp.complex64))
    _, result, _ = jax.lax.while_loop(cond, body, init)
    return result


def hyp2f1_recursive(a, b, c, z, max_terms=100, max_depth=5, verbose=True):
    def _inner(a, b, c, z, depth, prefactor=1.0):
        abs_z = abs(z)

        if abs_z < 1:
            if verbose:
                print("Converging: using Taylor series at", z)
            return prefactor * hyp2f1_taylor(a, b, c, z, max_terms=max_terms)

        elif depth > 0:
            if verbose:
                print(f"Attempting Pfaff (depth = {depth})")

            # Apply Pfaff transformation
            z_new = z / (z - 1)
            abs_new = abs(z_new)

            if verbose:
                print(f"Pfaff: |z| = {abs_z:.4f} → |z_new| = {abs_new:.4f}")

            # Compose the prefactor
            new_prefactor = prefactor * (1 - z) ** (-a)

            # Try again with transformed z
            return _inner(a, c - b, c, z_new, depth - 1, new_prefactor)

        else:
            if verbose:
                print(f"Switching to Euler at z = {z} after failed Pfaff attempts")
            euler_prefactor = prefactor * (1 - z) ** (c - a - b)
            return euler_prefactor * hyp2f1_taylor(c - a, c - b, c, z, max_terms=max_terms)

    return _inner(a, b, c, z, max_depth)



@jax.jit
def hyp2f1_terminating(a, b, c, z):
    n_max = jnp.minimum(-a, -b)
    
    def term(n):
        num = pochhammer_safe(a, n) * pochhammer_safe(b, n)
        denom = pochhammer_safe(c, n) * jnp.exp(gammaln(n + 1))
        return (num / denom) * z**n

    # Define things for jax.lax.while_loop

    def sum_terms_whileLoop(term, n_max):

        def cond_fun(state):
            n, acc = state 
            return n <= n_max

        def body_fun(state):
            n, acc = state
            acc = acc + term(n)
            return (n+1, acc)
        
        # initialize state 
        init_state = (0,0.0)
        _, total = jax.lax.while_loop(cond_fun, body_fun, init_state)

        return total 

    return sum_terms_whileLoop(term, n_max)

'''
Wrapper function for the Hyp2F1 function
'''

def hyp2f1(a, b, c, z, max_terms=100):
    mag_z = jnp.abs(z)
    is_integer_a = jnp.equal(a, jnp.floor(a))
    is_integer_b = jnp.equal(b, jnp.floor(b))

    is_terminating = (
        (is_integer_a and a <= 0) or
        (is_integer_b and b <= 0)
    )

    if is_terminating:
        return hyp2f1_terminating(a, b, c, z)
    elif mag_z < 1:
        return hyp2f1_taylor(a, b, c, z, max_terms=max_terms)
    else:
        return hyp2f1_recursive(a, b, c, z, max_terms=max_terms)

if __name__ == '__main__':

    print("Example case")
    a = -2
    b = -3
    c = a + b + 2
    z = 1.2 + 0.5j  # outside the unit disk

    _ = hyp2f1(a, b, c, z, max_terms = 100) # For JIT initialization

    start_time = time.time()
    value = hyp2f1(a, b, c, z, max_terms = 1000)
    print(value)
    print(f"{time.time() - start_time}")

    print("Compare with Scipy")
    start_time = time.time()
    
    print(special.hyp2f1(a, b, c, z))
    print(f"{time.time() - start_time}")








    







