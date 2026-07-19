"""
A linear transformation example written by Gemini Flash
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Create a grid where every dot falls strictly on a whole number
# This creates integers: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
range_values = np.arange(-4, 5, 1)
x, y = np.meshgrid(range_values, range_values)
grid = np.vstack([x.flatten(), y.flatten()])

# 2. Define your transformation matrix
# Let's do a shear: i_hat stays at [1, 0], j_hat moves to [1, 1]
A = np.array([[1.0, 0.5],
              [0.0, 1.0]])

# 3. Apply the transformation (matrix multiplication)
transformed_grid = A @ grid

# 4. Set up the plotting window
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ticks = np.arange(-6, 7, 1)  # Ensure grid lines extend far enough

for a in ax:
    a.set_xlim(-6, 6)
    a.set_ylim(-6, 6)
    a.set_xticks(ticks)
    a.set_yticks(ticks)
    
    # Clean whole-number grid lines
    a.axhline(0, color='black', lw=1.5, zorder=1)
    a.axvline(0, color='black', lw=1.5, zorder=1)
    a.grid(True, which='both', linestyle='--', color='gray', alpha=0.5, zorder=0)

# --- LEFT PLOT: Original Space ---
ax[0].set_title("Original Space (Whole Numbers Only)", fontsize=12)
ax[0].scatter(grid[0], grid[1], color='blue', alpha=0.6, edgecolors='k', s=40, zorder=3)

# Draw original basis vectors (i_hat = green, j_hat = red)
ax[0].quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label=r'$\hat{i}$ [1, 0]', zorder=4)
ax[0].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='red', label=r'$\hat{j}$ [0, 1]', zorder=4)
ax[0].legend(loc="upper left")

# --- RIGHT PLOT: Transformed Space ---
det_A = np.linalg.det(A)
ax[1].set_title(f"Transformed Space (Det = {det_A:.1f})", fontsize=12)
ax[1].scatter(transformed_grid[0], transformed_grid[1], color='magenta', alpha=0.6, edgecolors='k', s=40, zorder=3)

# Draw transformed basis vectors (columns of matrix A)
ax[1].quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1, color='green', label=f"Transformed $\hat{{i}}$ [{A[0,0]:.0f}, {A[1,0]:.0f}]", zorder=4)
ax[1].quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1, color='red', label=f"Transformed $\hat{{j}}$ [{A[0,1]:.0f}, {A[1,1]:.0f}]", zorder=4)
ax[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig('learning_scripts/learning_results/linear_transformation.png')