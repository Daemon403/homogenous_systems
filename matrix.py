import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import null_space

def solve_homogeneous(A):
    """Solve Ax = 0 and return basis for null space."""
    return null_space(A)

def solve_nonhomogeneous(A, b):
    """Solve Ax = b and return particular solution and null space (if exists)."""
    try:
        x_p = np.linalg.lstsq(A, b, rcond=None)[0]
        residual = A @ x_p - b
        if np.allclose(residual, 0):
            x_h = null_space(A)
            return x_p, x_h
        else:
            return None, None
    except np.linalg.LinAlgError:
        return None, None

def plot_equations_and_solutions(A, b, x_p, x_h):
    if A.shape[1] != 3:
        print("Plotting only works in 3D (3 variables).")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("System of Linear Equations: Planes, Vectors, and Solution Set")

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    equations = []
    xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))

    for i in range(A.shape[0]):
        a, b_coeff, c = A[i]
        d = b[i]
        if abs(c) > 1e-6:
            zz = (d - a * xx - b_coeff * yy) / c
        else:
            zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color=colors[i % len(colors)])
        equations.append(f"{a}x + {b_coeff}y + {c}z = {d}")

    for i, eq in enumerate(equations):
        ax.text2D(0.05, 0.95 - i * 0.05, f"Plane {i+1}: {eq}", transform=ax.transAxes)

    ax.scatter(0, 0, 0, color='black', s=50, label="Origin")

    if x_p is not None:
        ax.quiver(0, 0, 0, x_p[0], x_p[1], x_p[2], color='magenta', linewidth=2, label='Particular Vector P')
        ax.scatter(*x_p, color='magenta', s=40)

    if x_h is not None and x_h.size > 0:
        for i in range(x_h.shape[1]):
            v = x_h[:, i]
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color='black', linewidth=2, label=f'Null Vector V{i+1}')

    if x_p is not None and x_h is not None and x_h.size > 0:
        if x_h.shape[1] == 1:
            v = x_h[:, 0]
            t_vals = np.linspace(-10, 10, 100)
            x = x_p[0] + t_vals * v[0]
            y = x_p[1] + t_vals * v[1]
            z = x_p[2] + t_vals * v[2]
            ax.plot(x, y, z, color='purple', linewidth=2, label='P + span(V)')
        elif x_h.shape[1] == 2:
            v1 = x_h[:, 0]
            v2 = x_h[:, 1]
            s, t = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
            x = x_p[0] + s * v1[0] + t * v2[0]
            y = x_p[1] + s * v1[1] + t * v2[1]
            z = x_p[2] + s * v1[2] + t * v2[2]
            ax.plot_surface(x, y, z, alpha=0.4, color='purple', label='P + span(V)')

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    A = np.array([
        [2, -1, 1],
        [1, 1, 1]
    ])
    b = np.array([1, 2])

    # Solve homogeneous system
    homogeneous_solution = solve_homogeneous(A)
    print("\nSolution for Ax = 0:")
    if homogeneous_solution is None or homogeneous_solution.size == 0:
        print("Trivial solution: x = 0")
    else:
        print("Basis for the null space:")
        for i in range(homogeneous_solution.shape[1]):
            print(f"Vector {i+1}: {homogeneous_solution[:, i]}")

    # Solve nonhomogeneous system
    x_p, x_h = solve_nonhomogeneous(A, b)
    print("\nSolution for Ax = b:")
    if x_p is None:
        print("No solution (inconsistent system).")
    else:
        print("Particular solution:", x_p)
        if x_h is not None and x_h.size > 0:
            print("General solution: x = particular + span of homogeneous solutions")
        else:
            print("Unique solution (no homogeneous component).")

    # Plot if in R^3
    if A.shape[1] == 3:
        plot_equations_and_solutions(A, b, x_p, x_h)
    else:
        print("Plotting skipped: System is not in R^3.")