"""
A Systems of Equations Visualization 

by Gemini Flash
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP THE MATH PROBLEM
# =====================================================================
# The Transformation Matrix (A shear + scale)
A = np.array([[1.5, 1.0],
              [0.0, 1.0]])

# The Target Vector 'b' where we want to land in the transformed space
b = np.array([2.0, 3.0])

# Compute the Inverse Matrix to "rewind" space
A_inv = np.linalg.inv(A)

# Calculate the solution vector 'x' (Ax = b  ->  x = A^-1 * b)
x_sol = A_inv @ b

print(f"Target vector b: {b}")
print(f"Calculated solution vector x: {x_sol}")
print(f"Verification (A @ x): {A @ x_sol}")

# =====================================================================
# 2. GENERATE COORDINATE MESHES
# =====================================================================
# Create a standard whole-number grid for background reference
range_vals = np.arange(-5, 6, 1)
x_coords, y_coords = np.meshgrid(range_vals, range_vals)
grid = np.vstack([x_coords.flatten(), y_coords.flatten()])

# Transform the whole grid using matrix A
transformed_grid = A @ grid

# =====================================================================
# 3. VISUALIZATION PIPELINE
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ticks = np.arange(-7, 8, 1)

def apply_base_styling(axis, title):
    axis.set_xlim(-7, 7)
    axis.set_ylim(-7, 7)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.axhline(0, color='black', lw=1.5, zorder=1)
    axis.axvline(0, color='black', lw=1.5, zorder=1)
    axis.grid(True, which='both', linestyle='--', color='gray', alpha=0.4, zorder=0)
    axis.set_aspect('equal')
    axis.set_title(title, fontsize=12, pad=12)

# --- LEFT PLOT: Original Space (Finding the Input) ---
apply_base_styling(ax[0], "1. Original Space\nWhere does the vector start?")
# Plot the clean, unwarped whole-number grid points
ax[0].scatter(grid[0], grid[1], color='dodgerblue', alpha=0.4, s=20, zorder=2)

# Draw the calculated solution vector x
ax[0].quiver(0, 0, x_sol[0], x_sol[1], angles='xy', scale_units='xy', scale=1, 
             color='green', width=0.012, zorder=4, label=f'Solution $\\vec{{x}}$\n[{x_sol[0]:.2f}, {x_sol[1]:.2f}]')
ax[0].legend(loc="upper left", fontsize=10)

# --- RIGHT PLOT: Transformed Space (Landing on the Target) ---
apply_base_styling(ax[1], "2. Transformed Space ($A\\vec{x}$)\nWhere the vector lands.")
# Plot the warped grid points
ax[1].scatter(transformed_grid[0], transformed_grid[1], color='magenta', alpha=0.4, s=20, zorder=2)

# Draw the target vector b that we were trying to hit
ax[1].quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, 
             color='red', width=0.012, zorder=4, label=f'Target $\\vec{{b}}$\n[{b[0]:.1f}, {b[1]:.1f}]')

# Draw the transformed basis vectors to show how the grid warped
ax[1].quiver(0, 0, A[0,0], A[1,0], angles='xy', scale_units='xy', scale=1, color='darkgreen', alpha=0.8, linestyle='-', zorder=3)
ax[1].quiver(0, 0, A[0,1], A[1,1], angles='xy', scale_units='xy', scale=1, color='darkred', alpha=0.8, linestyle='-', zorder=3)
ax[1].legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig('learning_scripts/learning_results/systems.png')