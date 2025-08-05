import numpy as np

# Defining the bits as numpy arrays
bit0 = np.array([1, 0])  # |0>
bit1 = np.array([0, 1])  # |1>

# Definind some single qubit gates
def hadamard(bit):
    """Hadamard gate"""
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]).dot(bit)

def pauli_x(bit):
    """Pauli-X gate (bit flip)"""
    return np.array([[0, 1], [1, 0]]).dot(bit)

def pauli_y(bit):
    """Pauli-Y gate (bit flip and phase flip)"""
    return np.array([[0, -1j], [1j, 0]]).dot(bit)

def pauli_z(bit):
    """Pauli-Z gate (phase flip)"""
    return np.array([[1, 0], [0, -1]]).dot(bit)

