import numpy as np
from scipy.integrate import solve_bvp

# Constants
L = 4  # interval length
K = 1  # normalized constant
xspan = np.arange(-L, L + 0.1, 0.1)  # x span from -L to L with step of 0.1

# Define the boundary value problem for phi_n''(x) = (K * x^2 - epsilon_n) * phi_n(x)
def harmonic_oscillator(x, y, epsilon):
    return np.vstack((y[1], (K * x**2 - epsilon) * y[0]))

# Boundary conditions: phi(-L) = phi(L) = 0
def boundary_conditions(ya, yb):
    return np.array([ya[0], yb[0]])

# Function to solve the BVP for a given epsilon
def solve_for_eigenvalue(epsilon_guess):
    # Initial guess for y: [phi_n(x), phi_n'(x)]
    y_init = np.zeros((2, len(xspan)))
    y_init[0] = np.exp(-xspan**2)  # Initial guess for phi_n(x)

    # Solve the boundary value problem
    solution = solve_bvp(lambda x, y: harmonic_oscillator(x, y, epsilon_guess),
                         boundary_conditions, xspan, y_init)
    
    return solution.sol(xspan)[0], solution.success

# Placeholder for eigenfunctions and eigenvalues
eigenfunctions = []
eigenvalues = []

# Loop through the first 5 eigenvalues to calculate eigenfunctions
for epsilon_guess in np.array([0.5, 1.5, 2.5, 3.5, 4.5]):
    success = False
    while not success:
        phi_n, success = solve_for_eigenvalue(epsilon_guess)
        epsilon_guess += 0.01  # Adjust the guess slightly to converge

    # Normalize the eigenfunction
    norm = np.sqrt(np.trapz(phi_n**2, xspan))  # Normalization factor
    phi_n /= norm  # Normalize

    # Save the absolute values of eigenfunctions and eigenvalues
    eigenfunctions.append(np.abs(phi_n))  # Absolute value
    eigenvalues.append(epsilon_guess)  # Save the eigenvalue

# Convert the eigenfunctions list to a matrix and eigenvalues to a vector
eigenfunctions_matrix = np.column_stack(eigenfunctions)  # 5-column matrix
eigenvalues_vector = np.array(eigenvalues)  # 1x5 vector

# Print the results
print("Eigenfunctions (A1):")
print(eigenfunctions_matrix)
print("\nEigenvalues (A2):")
print(eigenvalues_vector)
