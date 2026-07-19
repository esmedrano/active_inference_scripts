"""
A Covariance and Precision Visualization 

by Gemini Flash
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a uniform sphere of raw, standard points (zero mean, unit variance)
np.random.seed(42)
num_points = 500
angles = np.random.uniform(0, 2*np.pi, num_points)
radii = np.sqrt(np.random.uniform(0, 4, num_points))  # Radius up to 2
raw_data = np.vstack([radii * np.cos(angles), radii * np.sin(angles)])

# 2. Define a transformation matrix to inject variance and correlation
# This creates a stretched, tilted ellipse of noise
L = np.array([[2.0, 0.5],
              [1.0, 1.5]])

# The resulting Covariance Matrix (Sigma = L @ L^T)
Sigma = L @ L.T
# The resulting Precision Matrix (Inverse of Covariance)
Precision = np.linalg.inv(Sigma)

# 3. Transform the uniform data into a correlated noise distribution
correlated_data = L @ raw_data

# 4. Use the Precision Matrix to whiten the data back to its standardized space
# Mathematically: We take the symmetric square root of the precision matrix (Pi^0.5)
# to linearly un-stretch the coordinates uniformly.
evals, evecs = np.linalg.eigh(Precision)
precision_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
whitened_data = precision_sqrt @ correlated_data

# =====================================================================
# PLOTTING THE SPATIAL TRANSFORMATION PIPELINE
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ticks = np.arange(-6, 7, 2)

titles = [
    "1. Standard Uniform Space\n(Unit Variance)",
    "2. Stretched Space via Covariance ($\Sigma$)\n(Correlated Noise Ellipse)",
    "3. Compressed Space via Precision ($\Pi$)\n(Returned to Normalized Spherical Distance)"
]
datasets = [raw_data, correlated_data, whitened_data]
colors = ['dodgerblue', 'crimson', 'mediumpurple']

for i in range(3):
    ax[i].set_xlim(-6, 6)
    ax[i].set_ylim(-6, 6)
    ax[i].set_xticks(ticks)
    ax[i].set_yticks(ticks)
    ax[i].axhline(0, color='black', lw=1.2, zorder=1)
    ax[i].axvline(0, color='black', lw=1.2, zorder=1)
    ax[i].grid(True, linestyle='--', color='gray', alpha=0.4, zorder=0)
    ax[i].set_aspect('equal')
    ax[i].set_title(titles[i], fontsize=11, pad=10)
    ax[i].scatter(datasets[i][0], datasets[i][1], color=colors[i], alpha=0.6, edgecolors='k', s=25, zorder=3)

plt.tight_layout()
plt.savefig('learning_scripts/learning_results/covariance.png')