import numpy as np

# Defining the bits as numpy arrays
bit0 = np.array([1, 0])  # |0>
bit1 = np.array([0, 1])  # |1>

# Definind some single qubit gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
X = np.array([[0, 1], [1, 0]])  # Pauli-X gate
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z gate
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T gate
S = np.array([[1, 0], [0, 1j]])  # S gate
I = np.array([[1, 0], [0, 1]])  # Identity gate

# So running a circuit with H, T, X with a qubit initialized to |0> will result in:
print(X @ T @ H @ bit0)


### 2 qubit 
zerozero = np.kron(bit0, bit0)  # |00>
zerone = np.kron(bit0, bit1)  # |01>
onezero = np.kron(bit1, bit0)  # |10>
oneone = np.kron(bit1, bit1)  # |11>

# Defining some two qubit gates
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])  # CNOT gate. Control is the first qubit (bottom), target is the second qubit (top)

CNOT2 = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]])  # CNOT gate with control and target swapped

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]])  # Controlled-Z gate

# If I apply H on both qubits and a CNOT gate, I will get the following state:
print(CNOT @ (np.kron(H @ bit0, H @ bit0)))