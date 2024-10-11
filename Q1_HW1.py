import numpy as np

# Define the function and its derivative
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

# Newton-Raphson Method
x = np.array([-1.6])  # Initial guess
tolerance = 1e-6
max_iterations = 1000

for j in range(max_iterations):
    next_x = x[j] - f(x[j]) / df(x[j])
    x = np.append(x, next_x)
    
    # Check convergence with f(x)
    if abs(f(x[j])) < tolerance:
        A1 = x  # A1 as the vector of x-values for Newton-Raphson
        newton_iterations = j + 1
        break
else:
    print("Newton-Raphson Method did not converge")

# Bisection Method
xl, xr = -0.7, -0.4  # Initial endpoints
A2 = []  # Store midpoint values
for j in range(max_iterations):
    xc = (xl + xr) / 2
    A2.append(xc)
    
    # Update bounds based on sign of f(xc)
    if f(xc) > 0:
        xl = xc
    else:
        xr = xc
    
    # Check convergence with f(xc)
    if abs(f(xc)) < tolerance:
        bisection_iterations = j + 1
        break
else:
    print("Bisection Method did not converge")

A2 = np.array(A2)  # Convert to numpy array for consistency
A3 = np.array([newton_iterations, bisection_iterations])  # Iteration counts

# Print results
print("A1 (Newton-Raphson x-values):", A1)
print("A2 (Bisection midpoints):", A2)
print("A3 (Number of iterations):", A3)
import numpy as np

# Define the matrices and vectors
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

# Calculate the required operations
A4 = A + B                 # (a) A + B
A5 = 3 * x - 4 * y         # (b) 3x - 4y
A6 = A @ x                 # (c) Ax
A7 = B @ (x - y)           # (d) B(x - y)
A8 = D @ x                 # (e) D x
A9 = D @ y + z             # (f) D y + z
A10 = A @ B                # (g) AB
A11 = B @ C                # (h) BC
A12 = C @ D                # (i) CD

# Print results
print("A4 (A + B):\n", A4)
print("A5 (3x - 4y):\n", A5)
print("A6 (Ax):\n", A6)
print("A7 (B(x - y)):\n", A7)
print("A8 (D x):\n", A8)
print("A9 (D y + z):\n", A9)
print("A10 (AB):\n", A10)
print("A11 (BC):\n", A11)
print("A12 (CD):\n", A12)
