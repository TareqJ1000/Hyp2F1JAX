import numpy as np
import matplotlib.pyplot as plt
from hyp2f1_jax import hyp2f1  # Use the wrapper
from scipy.special import hyp2f1 as scipy_hyp2f1
import jax
import jax.numpy as jnp
import os

# Parameters for the hypergeometric function (all integers)
# a = 2
# b = 3
# c = 5

# Generate 1000 random complex numbers outside the unit disc (1 < |z| < 100)
np.random.seed(42)
r_out = np.sqrt(np.random.uniform(1, 100**2, 1000))
theta_out = np.random.uniform(0, 2 * np.pi, 1000)
z = r_out * np.exp(1j * theta_out)

all_abs_diff = []
a_range = range(0, -11, -1)  # 0, -1, ..., -10
b_range = range(0, -11, -1)  # 0, -1, ..., -10
n_range = range(0, 21)       # 0, 1, ..., 20

total_combos = len(a_range) * len(b_range) * len(n_range)
combo_count = 0

scipy_inf_abc_set = set()
scipy_inf_abc_file = 'scipy_inf_abc_cases.txt'

# Load previous (a, b, c) inf cases if file exists
if os.path.exists(scipy_inf_abc_file):
    with open(scipy_inf_abc_file, 'r') as f:
        for line in f:
            try:
                a, b, c = map(int, line.strip().split(','))
                scipy_inf_abc_set.add((a, b, c))
            except Exception as e:
                print(f"Could not parse line in {scipy_inf_abc_file}: {line}\nError: {e}")

new_scipy_inf_abc = set()

for a in a_range:
    for b in b_range:
        for n in n_range:
            c = a + b + abs(n)
            combo_count += 1
            if (a, b, c) in scipy_inf_abc_set:
                print(f"Skipping (a={a}, b={b}, c={c}) due to previous inf.")
                continue
            print(f"\n===== Combo {combo_count}/{total_combos}: a={a}, b={b}, n={n}, c={c} =====")
            jax_results = []
            inf_found = False
            for i, zi in enumerate(z):
                #print(i)
                val = hyp2f1(a, b, c, zi, max_terms=1000)
                val_np = np.asarray(val)
                val_scipy = scipy_hyp2f1(a, b, c, zi)
                if np.isinf(val_scipy):
                    print(f"Warning: SciPy returned inf! a={a}, b={b}, c={c}, z={zi}, abs(z)={np.abs(zi)}, scipy_hyp2f1={val_scipy}")
                    new_scipy_inf_abc.add((a, b, c))
                    jax_results.append(0)
                    inf_found = True
                    break  # No need to check further z for this (a, b, c)
                elif np.isnan(val_np):
                    print(f"Warning: NaN detected! a={a}, b={b}, c={c}, z={zi}, abs(z)={np.abs(zi)}. Using 0 instead.")
                    jax_results.append(0)
                    #input()
                else:
                    abs_diff = np.abs(val_np - val_scipy)
                    #print(f"Success :), a={a}, b={b}, c={c}, z={zi}, abs(z)={np.abs(zi)}, abs_diff={abs_diff}.")
                    #print(val_np)
                    #print(val_scipy)
                    jax_results.append(val_np)
                    #input()
            if inf_found:
                continue  # Skip the rest of this combo
            jax_results = np.array(jax_results)

            # Compute using scipy implementation
            scipy_results = scipy_hyp2f1(a, b, c, z)

            # Compute absolute differences
            abs_diff = np.abs(jax_results - scipy_results)
            all_abs_diff.extend(abs_diff)
            print(f"Combo {combo_count}: min abs_diff={abs_diff.min()}, max abs_diff={abs_diff.max()}, mean abs_diff={abs_diff.mean()}")
            #input()
all_abs_diff = np.array(all_abs_diff)
print(all_abs_diff)

# After all combos, save new (a, b, c) inf cases to file (append new ones)
if new_scipy_inf_abc:
    with open(scipy_inf_abc_file, 'a') as f:
        for abc in new_scipy_inf_abc:
            f.write(f"{abc[0]},{abc[1]},{abc[2]}\n")

# Create 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot histogram with logarithmic bins
plt.figure(figsize=(8, 5))
min_diff = all_abs_diff[all_abs_diff > 0].min() if np.any(all_abs_diff > 0) else 1e-16
max_diff = all_abs_diff.max()
bins = np.logspace(np.log10(min_diff), np.log10(max_diff), 50)
plt.hist(all_abs_diff, bins=bins, color='skyblue', edgecolor='black')
plt.xscale('log')
plt.title('Absolute Difference between JAX and SciPy hyp2f1 Implementations\n(0 >= a >= -10, 0 >= b >= -10, 0 <= n <= 20, c=a+b+|n|, 1000 z)')
plt.xlabel('Absolute Difference (log scale)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6, which='both')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'histogram_abs_diff_1000_examples_neg_a_b.png'), dpi=150)
plt.show() 