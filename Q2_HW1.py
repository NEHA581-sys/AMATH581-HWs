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
