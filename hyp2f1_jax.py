import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

# Safer Pochhammer (a)_n using gamma function
def pochhammer(a, n):
    return jnp.exp(gammaln(a + n) - gammaln(a))

# Taylor series for |z| < 1
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

    init = (0, jnp.zeros_like(z, dtype=z.dtype), jnp.ones_like(z, dtype=z.dtype))
    _, result, _ = jax.lax.while_loop(cond, body, init)
    return result
