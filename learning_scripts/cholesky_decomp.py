"""
A Cholesky Decompisition Visualization 

by Gemini Flash
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DEFINE TARGET CORRELATION & COVARIANCE
# =====================================================================
# Let's say we want a distribution where:
# - Variance of X variable = 4.0 (Standard Deviation = 2.0)
# - Variance of Y variable = 1.0 (Standard Deviation = 1.0)
# - They have a strong positive correlation/covariance of 1.5
Sigma = np.array([[4.0, 1.5],
                  [1.5, 1.0]])

# Compute the Cholesky Decomposition (Sigma = L @ L.T)
# L is a lower-triangular matrix acting as the spatial "shaper"
L = np.linalg.cholesky(Sigma)

print("Target Covariance Matrix (Sigma):")
print(Sigma)
print("\nCholesky Factor Matrix (L):")
print(L)

# =====================================================================
# 2. GENERATE UNCORRELATED WHITE NOISE
# =====================================================================
np.random.seed(42)
num_samples = 800
# 2 rows (X, Y), 'num_samples' columns of standard normal distribution
white_noise = np.random.normal(0, 1, size=(2, num_samples))

# =====================================================================
# 3. TRANSFORM WHITE NOISE VIA CHOLESKY FACTOR L
# =====================================================================
# This matrix multiplication applies the spatial stretch and shear
correlated_noise = L @ white_noise

# Verify the empirical covariance of our generated data matches Sigma
empirical_covariance = np.cov(correlated_noise)
print("\nEmpirical Covariance of Transformed Data:")
print(empirical_covariance)

# =====================================================================
# 4. VISUALIZATION PIPELINE
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ticks = np.arange(-6, 7, 1)

def apply_base_styling(axis, title):
    axis.set_xlim(-6, 6)
    axis.set_ylim(-6, 6)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.axhline(0, color='black', lw=1.5, zorder=1)
    axis.axvline(0, color='black', lw=1.5, zorder=1)
    axis.grid(True, which='both', linestyle='--', color='gray', alpha=0.4, zorder=0)
    axis.set_aspect('equal')
    axis.set_title(title, fontsize=12, pad=12)

# --- LEFT PLOT: Uncorrelated White Noise ---
apply_base_styling(ax[0], "1. Uncorrelated White Noise\n(Symmetrical Spheroid Map)")
ax[0].scatter(white_noise[0], white_noise[1], color='dodgerblue', alpha=0.6, edgecolors='k', s=25, zorder=3)

# Draw original independent basis components
ax[0].quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', width=0.01, zorder=4, label='Independent X variance')
ax[0].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='red', width=0.01, zorder=4, label='Independent Y variance')
ax[0].legend(loc="upper left")

# --- RIGHT PLOT: Transformed Correlated Noise ---
apply_base_styling(ax[1], "2. Correlated Noise Cloud\n(Warped via Cholesky Matrix $L$)")
ax[1].scatter(correlated_noise[0], correlated_noise[1], color='crimson', alpha=0.6, edgecolors='k', s=25, zorder=3)

# Draw the transformed basis columns of L to show the direction of spatial stretching
ax[1].quiver(0, 0, L[0,0], L[1,0], angles='xy', scale_units='xy', scale=1, color='green', width=0.01, zorder=4, label='Transformed Column 1')
ax[1].quiver(0, 0, L[0,1], L[1,1], angles='xy', scale_units='xy', scale=1, color='red', width=0.01, zorder=4, label='Transformed Column 2')
ax[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig('learning_scripts/learning_results/cholesky_decomp.png')