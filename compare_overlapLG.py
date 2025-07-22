import numpy as np
import matplotlib.pyplot as plt
import os
from OverlapLG import overlapIntegral as overlapLG_np
from OverlapLG_jax import overlapIntegral as overlapLG_jax

# Constants (match both implementations)
cm = 1e-2
mm = 1e-3
nm = 1e-9
lambda0 = 810 * nm
w0 = 1 * mm
k0 = 2 * np.pi / lambda0
zR0 = (np.pi * w0 ** 2) / lambda0
DELTA = 1e-10

# Parameter ranges
l_values = np.arange(-3, 4)
p_values = np.arange(0, 41)
p2_values = np.arange(0, 41)
z_range = (0 + DELTA, zR0 + DELTA)
z2_range = (0 + 2 * DELTA, zR0 + 2 * DELTA)
w_range = (1 * mm + DELTA, 1.1 * mm + DELTA)
w2_range = (1 * mm + 2 * DELTA, 1.1 * mm + 2 * DELTA)

n_examples = 100000
all_abs_diff = []
nan_cases = []
complex_exp_failures = 0
successful_experiments = 0
jax_failed_only = 0

np.random.seed(42)

for i in range(n_examples):
    l = int(np.random.choice(l_values))
    p = int(np.random.choice(p_values))
    p2 = int(np.random.choice(p2_values))
    z = np.random.uniform(*z_range)
    z2 = np.random.uniform(*z2_range)
    w = np.random.uniform(*w_range)
    w2 = np.random.uniform(*w2_range)
    print(f"\n===== Example {i+1}/{n_examples}: l={l}, p={p}, p2={p2}, z={z}, z2={z2}, w={w}, w2={w2} =====")
    try:
        val_np = overlapLG_np(l, p, p2, z, z2, w, w2)
    except Exception as e:
        print(f"NumPy/Scipy OverlapLG failed: {e}")
        val_np = np.nan
        if 'complex' in str(e).lower() and 'exponent' in str(e).lower():
            complex_exp_failures += 1
        continue
    try:
        val_jax = overlapLG_jax(l, p, p2, z, z2, w, w2)
        if hasattr(val_jax, 'block_until_ready'):
            val_jax = np.asarray(val_jax)
    except Exception as e:
        print(f"JAX OverlapLG failed: {e}")
        val_jax = np.nan
        if 'complex' in str(e).lower() and 'exponent' in str(e).lower():
            complex_exp_failures += 1
    if np.isnan(val_np) or np.isnan(val_jax):
        nan_cases.append((l, p, p2, z, z2, w, w2, val_np, val_jax))
        if not np.isnan(val_np) and np.isnan(val_jax):
            jax_failed_only += 1
        print(f"NaN encountered: l={l}, p={p}, p2={p2}, z={z}, z2={z2}, w={w}, w2={w2}, val_np={val_np}, val_jax={val_jax}")
        continue
    abs_diff = np.abs(val_np - val_jax)
    all_abs_diff.append(abs_diff)
    successful_experiments += 1
    print(f"Example {i+1}: |NumPy-SciPy - JAX| = {abs_diff}")

all_abs_diff = np.array(all_abs_diff)
print("All absolute differences:", all_abs_diff)
print(f"Number of examples failed due to complex exponentiation: {complex_exp_failures}")
print(f"Number of successful experiments: {successful_experiments}")
print(f"Number of times only JAX failed but SciPy succeeded: {jax_failed_only}")

# Save NaN cases to file
if nan_cases:
    with open('overlapLG_nan_cases.txt', 'w') as f:
        for case in nan_cases:
            l, p, p2, z, z2, w, w2, val_np, val_jax = case
            f.write(f"l={l}, p={p}, p2={p2}, z={z}, z2={z2}, w={w}, w2={w2}, val_np={val_np}, val_jax={val_jax}\n")

# Create 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot histogram with logarithmic bins (skip NaN cases)
if np.any(all_abs_diff > 0):
    plt.figure(figsize=(8, 5))
    min_diff = all_abs_diff[all_abs_diff > 0].min() if np.any(all_abs_diff > 0) else 1e-16
    max_diff = all_abs_diff.max()
    bins = np.logspace(np.log10(min_diff), np.log10(max_diff), 50)
    plt.hist(all_abs_diff, bins=bins, color='salmon', edgecolor='black')
    plt.xscale('log')
    plt.title(f'Absolute Difference between NumPy/SciPy and JAX OverlapLG Implementations\n({successful_experiments} Random Experimental LG Parameters, p,p2 up to 40)')
    plt.xlabel('Absolute Difference (log scale)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'histogram_abs_diff_overlapLG_10000.png'), dpi=150)
    plt.show()
else:
    print("No valid (non-NaN) results to plot.") 