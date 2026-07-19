"""
Matrix Operations and their Geometric Visulaizations 

by Gemini Flash
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# HELPER FUNCTION: Setup a clean whole-number grid for plotting
# =====================================================================
def create_base_grid(min_val=-4, max_val=5):
    """Creates a grid of coordinates falling strictly on whole numbers."""
    range_values = np.arange(min_val, max_val, 1)
    x, y = np.meshgrid(range_values, range_values)
    return np.vstack([x.flatten(), y.flatten()])

def setup_axes(ax, title, limits=[-6, 6]):
    """Applies uniform styling with a grid line for every whole number."""
    ticks = np.arange(limits[0], limits[1] + 1, 1)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.axhline(0, color='black', lw=1.5, zorder=1)
    ax.axvline(0, color='black', lw=1.5, zorder=1)
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.4, zorder=0)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_aspect('equal')

# =====================================================================
# 1. MATRIX-VECTOR MULTIPLICATION (Transforming a Point)
# =====================================================================
def demo_matrix_vector():
    """
    Geometric Meaning: Moving a single specific vector to its new location
    in the morphed space dictated by the basis landing pads.
    """
    grid = create_base_grid()
    # Rotation matrix (approx 45 degrees)
    theta = np.radians(45)
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # Specific vector to track
    v = np.array([2, 3])
    v_transformed = A @ v
    grid_transformed = A @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    setup_axes(ax[0], "Matrix-Vector: Original Vector v=[2,3]")
    ax[0].scatter(grid[0], grid[1], color='blue', alpha=0.3, s=15)
    ax[0].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='darkorange', width=0.015)
    
    setup_axes(ax[1], "Matrix-Vector: Transformed Vector (Rotated 45°)")
    ax[1].scatter(grid_transformed[0], grid_transformed[1], color='magenta', alpha=0.3, s=15)
    ax[1].quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='darkorange', width=0.015)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/matrix_vector.png')

# =====================================================================
# 2. MATRIX-MATRIX MULTIPLICATION (Composition of Transformations)
# =====================================================================
def demo_matrix_multiplication():
    """
    Geometric Meaning: Applying sequential space transformations.
    B happens first (Shear), then A happens second (Reflection).
    """
    grid = create_base_grid()
    
    B = np.array([[1.0, 1.5],   # Horizontal Shear
                  [0.0, 1.0]])
                  
    A = np.array([[1.0, 0.0],   # Vertical Reflection across X-axis
                  [0.0, -1.0]])
    
    grid_after_B = B @ grid
    grid_after_AB = A @ grid_after_B  # Composition: A @ B @ grid

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    setup_axes(ax[0], "1. Original Identity Grid")
    ax[0].scatter(grid[0], grid[1], color='blue', alpha=0.5, s=15)
    
    setup_axes(ax[1], "2. After Matrix B (Horizontal Shear)")
    ax[1].scatter(grid_after_B[0], grid_after_B[1], color='orange', alpha=0.5, s=15)
    
    setup_axes(ax[2], "3. After Matrix A (Reflected over X-Axis)")
    ax[2].scatter(grid_after_AB[0], grid_after_AB[1], color='magenta', alpha=0.5, s=15)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/matrix_matrix.png')

# =====================================================================
# 3. MATRIX ADDITION (Combining Vector Fields / Shifts)
# =====================================================================
def demo_matrix_addition():
    """
    Geometric Meaning: Applying two independent spatial deformations 
    simultaneously (like adding two distinct physical forces acting on space).
    """
    grid = create_base_grid()
    
    A = np.array([[0.0, 1.0],   # Shear force
                  [0.0, 0.0]])
                  
    B = np.array([[1.0, 0.0],   # Scaling force
                  [0.0, 0.5]])
    
    grid_A = A @ grid
    grid_B = B @ grid
    grid_combined = (A + B) @ grid

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    setup_axes(ax[0], "Transformation A (Pure Shear)")
    ax[0].scatter(grid_A[0], grid_A[1], color='teal', alpha=0.5, s=15)
    
    setup_axes(ax[1], "Transformation B (Pure Scaling)")
    ax[1].scatter(grid_B[0], grid_B[1], color='coral', alpha=0.5, s=15)
    
    setup_axes(ax[2], "Combined Transformation (A + B)")
    ax[2].scatter(grid_combined[0], grid_combined[1], color='magenta', alpha=0.5, s=15)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/matrix_addition.png')

# =====================================================================
# 4. SCALAR MULTIPLICATION (Uniform Scaling)
# =====================================================================
def demo_scalar_multiplication():
    """
    Geometric Meaning: Scaling the entire resulting output space uniformly,
    stretching or shrinking all distances from the origin by constant c.
    """
    grid = create_base_grid()
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    
    c = 2.0
    grid_transformed = A @ grid
    grid_scaled = (c * A) @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    setup_axes(ax[0], "Standard Morph (Matrix A)")
    ax[0].scatter(grid_transformed[0], grid_transformed[1], color='blue', alpha=0.5, s=15)
    
    setup_axes(ax[1], f"Uniformly Scaled Morph (Constant c = {c})")
    ax[1].scatter(grid_scaled[0], grid_scaled[1], color='magenta', alpha=0.5, s=15)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/scalar_multiplication.png')

# =====================================================================
# 5. MATRIX TRANSPOSE (Reflecting Axis System)
# =====================================================================
def demo_matrix_transpose():
    """
    Geometric Meaning: Swapping the roles of the basis vectors, effectively
    reflecting the spatial manipulation across the main diagonal axis.
    """
    grid = create_base_grid()
    A = np.array([[2.0, 1.0],
                  [0.5, 1.0]])
    
    grid_A = A @ grid
    grid_AT = A.T @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    setup_axes(ax[0], "Original Matrix Transformation A")
    ax[0].scatter(grid_A[0], grid_A[1], color='blue', alpha=0.5, s=15)
    # Plot basis vectors as reference arrows
    ax[0].quiver(0, 0, A[0,0], A[1,0], angles='xy', scale_units='xy', scale=1, color='green', zorder=4)
    ax[0].quiver(0, 0, A[0,1], A[1,1], angles='xy', scale_units='xy', scale=1, color='red', zorder=4)
    
    setup_axes(ax[1], "Transposed Matrix Transformation A^T")
    ax[1].scatter(grid_AT[0], grid_AT[1], color='magenta', alpha=0.5, s=15)
    # Notice the arrows have swapped their x/y components
    ax[1].quiver(0, 0, A.T[0,0], A.T[1,0], angles='xy', scale_units='xy', scale=1, color='green', zorder=4)
    ax[1].quiver(0, 0, A.T[0,1], A.T[1,1], angles='xy', scale_units='xy', scale=1, color='red', zorder=4)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/transpose.png')

# =====================================================================
# 6. DETERMINANT (Volume/Area Scaling Factor & Space Collapsing)
# =====================================================================
def demo_determinant():
    """
    Geometric Meaning: The factor by which the area of a unit square changes.
    If det = 0, space loses a dimension completely and collapses to a line/point.
    """
    grid = create_base_grid()
    
    # Non-singular matrix (det != 0)
    A_stable = np.array([[2.0, 0.0],
                         [0.0, 1.5]])
    
    # Singular matrix (det == 0) -> Second column is just 2x the first column
    A_collapse = np.array([[1.0, 2.0],
                           [2.0, 4.0]])
    
    grid_stable = A_stable @ grid
    grid_collapse = A_collapse @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    det_s = np.linalg.det(A_stable)
    setup_axes(ax[0], f"Space Expanded (Det = {det_s:.2f})")
    ax[0].scatter(grid_stable[0], grid_stable[1], color='blue', alpha=0.5, s=15)
    # Highlight unit area box
    ax[0].fill([0, A_stable[0,0], A_stable[0,0]+A_stable[0,1], A_stable[0,1]], 
               [0, A_stable[1,0], A_stable[1,0]+A_stable[1,1], A_stable[1,1]], color='yellow', alpha=0.4, label='Area Factor')
    
    det_c = np.linalg.det(A_collapse)
    setup_axes(ax[1], f"Space Collapsed into 1D Line (Det = {det_c:.2f})")
    ax[1].scatter(grid_collapse[0], grid_collapse[1], color='red', alpha=0.6, s=15)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/determinant.png')

# =====================================================================
# 7. INVERSE (Undoing the Morph / Solving Systems)
# =====================================================================
def demo_inverse():
    """
    Geometric Meaning: Rewinding space back to its original configuration.
    Also shows solving A*x = b geometrically by finding the original point x.
    """
    grid = create_base_grid()
    A = np.array([[1.0, 1.5],
                  [0.0, 1.0]])
    A_inv = np.linalg.inv(A)
    
    # Target vector b in transformed space, and solving for x
    b = np.array([4, 2])
    x_solution = A_inv @ b
    
    grid_transformed = A @ grid
    grid_undone = A_inv @ grid_transformed

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    setup_axes(ax[0], "1. Original Grid with solution point x")
    ax[0].scatter(grid[0], grid[1], color='blue', alpha=0.3, s=15)
    ax[0].plot(x_solution[0], x_solution[1], 'go', markersize=8, label='Original x')
    ax[0].legend()
    
    setup_axes(ax[1], "2. Transformed Grid: Point x lands on Target b")
    ax[1].scatter(grid_transformed[0], grid_transformed[1], color='orange', alpha=0.3, s=15)
    ax[1].plot(b[0], b[1], 'ro', markersize=8, label='Target b')
    ax[1].legend()
    
    setup_axes(ax[2], "3. Inverse Applied: Returned to Identity Layout")
    ax[2].scatter(grid_undone[0], grid_undone[1], color='magenta', alpha=0.3, s=15)
    ax[2].plot(x_solution[0], x_solution[1], 'go', markersize=8)
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/inverse.png')

# =====================================================================
# 8. EIGENVALUES & EIGENVECTORS (Axes of Pure Scaling)
# =====================================================================
def demo_eigen():
    """
    Geometric Meaning: Finding lines in space that do not change direction
    when space morphs; vectors along these axes only scale by magnitude lambda.
    """
    grid = create_base_grid()
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    grid_transformed = A @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Prepare lines along eigenvector directions
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    t_vals = np.linspace(-6, 6, 100)
    
    for i, a in enumerate(ax):
        g = grid if i == 0 else grid_transformed
        title = "Original Space + Eigen-Axes" if i == 0 else "Transformed Space: Axes Only Stretched"
        setup_axes(a, title)
        a.scatter(g[0], g[1], color='blue' if i==0 else 'magenta', alpha=0.3, s=15)
        
        # Plot infinite tracking lines for eigenvectors
        a.plot(t_vals * v1[0], t_vals * v1[1], color='green', linestyle=':', label=f'Axis 1 (λ={eigenvalues[0]:.1f})')
        a.plot(t_vals * v2[0], t_vals * v2[1], color='red', linestyle=':', label=f'Axis 2 (λ={eigenvalues[1]:.1f})')
        a.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/eigen_values_vectors.png')

# =====================================================================
# 9. OUTER PRODUCT (Projecting Space onto a Line)
# =====================================================================
def demo_outer_product():
    """
    Geometric Meaning: Multiplying a column by a row compresses the entire
    2D coordinate field down completely onto a specific 1D line vector footprint.
    """
    grid = create_base_grid()
    
    u = np.array([2, 1])   # Column direction vector
    v = np.array([1, 1])   # Row scaling vector
    
    # Outer product uv^T creates a rank-1 matrix
    A_outer = np.outer(u, v) 
    grid_transformed = A_outer @ grid

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    setup_axes(ax[0], "Original Open 2D Space")
    ax[0].scatter(grid[0], grid[1], color='blue', alpha=0.5, s=15)
    
    setup_axes(ax[1], "Transformed via Outer Product Matrix")
    ax[1].scatter(grid_transformed[0], grid_transformed[1], color='magenta', alpha=0.6, s=25, zorder=3)
    
    # Highlight the line of the output vector u that captured everything
    t_vals = np.linspace(-6, 6, 100)
    ax[1].plot(t_vals * u[0], t_vals * u[1], color='darkorange', linestyle='-', lw=2, label='Direction Vector u')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('learning_scripts/learning_results/matrix_ops/outer_product.png')

# =====================================================================
# MAIN EXECUTION ROUTINE
# =====================================================================
def main():
    # Toggle individual functions on or off by commenting/uncommenting below:
    
    print("Running Matrix-Vector Multiplication...")
    demo_matrix_vector()
    
    print("Running Matrix-Matrix Multiplication (Composition)...")
    demo_matrix_multiplication()
    
    print("Running Matrix Addition...")
    demo_matrix_addition()
    
    print("Running Scalar Multiplication...")
    demo_scalar_multiplication()
    
    print("Running Matrix Transpose...")
    demo_matrix_transpose()
    
    print("Running Determinant & Spatial Collapse...")
    demo_determinant()
    
    print("Running Matrix Inverse & System Solutions...")
    demo_inverse()
    
    print("Running Eigenvalues & Eigenvectors...")
    demo_eigen()
    
    print("Running Outer Product Mapping...")
    demo_outer_product()

if __name__ == "__main__":
    main()