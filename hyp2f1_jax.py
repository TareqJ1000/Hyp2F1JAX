import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax.scipy.special import gammaln, gamma
from scipy import special
import time
from jax.experimental.ode import odeint

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

    # Enforce complex128 dtype
    init = (
        0,
        jnp.zeros_like(z, dtype=jnp.complex128),
        jnp.ones_like(z, dtype=jnp.complex128)
    )
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



def hyp2f1_terminating(a, b, c, z):
    # Truncate before invalid terms in the denominator
    n_max = jnp.minimum(jnp.minimum(-a, -b), -c - 1)

    def term(n):
        num = pochhammer_safe(a, n) * pochhammer_safe(b, n)
        denom = pochhammer_safe(c, n) * jnp.exp(gammaln(n + 1))

        # Avoid division by zero or inf
        safe = jnp.logical_and(jnp.isfinite(denom), denom != 0.0)
        t = jnp.where(safe, (num / denom) * z**n, 0.0)
        return t.astype(jnp.complex128)

    def sum_terms_kahan(term, n_max):
        def cond_fun(state):
            n, acc, comp = state
            return n <= n_max

        def body_fun(state):
            n, acc, comp = state
            t = term(n)
            y = t - comp
            temp = acc + y
            comp = (temp - acc) - y
            acc = temp
            return (n + 1, acc, comp)

        init_state = (
            0,
            jnp.array(0.0 + 0.0j, dtype=jnp.complex128),  # acc
            jnp.array(0.0 + 0.0j, dtype=jnp.complex128),  # compensation
        )

        _, total, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return total

    return sum_terms_kahan(term, n_max)


'''
Wrapper function for the Hyp2F1 function
'''

@jax.jit
def hyp2f1(a, b, c, z, max_terms=100):
    is_a0 = jnp.equal(a, 0)
    is_b0 = jnp.equal(b, 0)
    is_identity = jnp.logical_or(is_a0, is_b0)

    is_integer_a = jnp.equal(a, jnp.floor(a))
    is_integer_b = jnp.equal(b, jnp.floor(b))
    is_terminating = (
        (is_integer_a & (a <= 0)) |
        (is_integer_b & (b <= 0))
    )

    def _identity(_):
        return jnp.array(1.0 + 0.0j, dtype=jnp.complex128)

    def _not_identity(_):
        def _taylor():
            return hyp2f1_taylor(a, b, c, z, max_terms=max_terms)
        def _terminating():
            return hyp2f1_terminating(a, b, c, z)
        return jax.lax.cond(
            is_terminating,
            lambda _: _terminating(),
            lambda _: _taylor(),
            operand=None
        )

    return jax.lax.cond(
        is_identity,
        _identity,
        _not_identity,
        operand=None
    )

if __name__ == '__main__':

    print("Example case")
    a = -9
    b = -8
    c = -3
    z = 1.2 + 0.5j  # outside the unit disk

    _ = hyp2f1(a, b, c, z, max_terms = 100) # For JIT initialization

    start_time = time.time()
    value = hyp2f1(a, b, c, z, max_terms = 1000)
    print(value)
    #print(f"{time.time() - start_time}")

    print("Compare with Scipy")
    start_time = time.time()
    
    print(special.hyp2f1(a, b, c, z))
    print(f"{time.time() - start_time}")








    







