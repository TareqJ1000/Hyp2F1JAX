import numpy as np
import matplotlib.pyplot as plt
from hyp2f1_jax import hyp2f1  # Use the wrapper
from scipy.special import hyp2f1 as scipy_hyp2f1
import jax
import jax.numpy as jnp

# Parameters for the hypergeometric function (all integers)
a = 2
b = 3
c = 5

# Generate 1000 random complex numbers outside the unit disc (1 < |z| < 100)
np.random.seed(42)
r_out = np.sqrt(np.random.uniform(1, 100**2, 1000))
theta_out = np.random.uniform(0, 2 * np.pi, 1000)
z = r_out * np.exp(1j * theta_out)

# Compute using JAX implementation (wrapper)
jax_results = []
for i, zi in enumerate(z):
    val = hyp2f1(a, b, c, zi, max_terms=500)
    val_np = np.asarray(val)
    if np.isnan(val_np):
        print(f"Warning: NaN detected! a={a}, b={b}, c={c}, z={zi}, abs(z)={np.abs(zi)}. Using 0 instead.")
        jax_results.append(0)
        input()
    else:
        abs_diff = np.abs(val_np - scipy_hyp2f1(a, b, c, zi))
        print(f"Success :), a={a}, b={b}, c={c}, z={zi}, abs(z)={np.abs(zi)}, abs_diff={abs_diff}.")
        jax_results.append(val_np)
jax_results = np.array(jax_results)

# Compute using scipy implementation
scipy_results = scipy_hyp2f1(a, b, c, z)

# Compute absolute differences
abs_diff = np.abs(jax_results - scipy_results)
print(abs_diff)

# Plot histogram with logarithmic bins
plt.figure(figsize=(8, 5))
min_diff = abs_diff[abs_diff > 0].min() if np.any(abs_diff > 0) else 1e-16
max_diff = abs_diff.max()
bins = np.logspace(np.log10(min_diff), np.log10(max_diff), 50)
plt.hist(abs_diff, bins=bins, color='skyblue', edgecolor='black')
plt.xscale('log')
plt.title('Absolute Difference between JAX and SciPy hyp2f1 Implementations (1000 examples, |z| < 1 and |z| > 1)')
plt.xlabel('Absolute Difference (log scale)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6, which='both')
plt.tight_layout()
plt.savefig(f'histogram_abs_diff_1000_examples.png', dpi=150)
plt.show() 