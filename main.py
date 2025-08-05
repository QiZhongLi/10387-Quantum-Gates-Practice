import numpy as np

# Defining the bits as numpy arrays
bit0 = np.array([1, 0])  # |0>
bit1 = np.array([0, 1])  # |1>

# Definind some single qubit gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
X = np.array([[0, 1], [1, 0]])  # Pauli-X gate
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z gate
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate



