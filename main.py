import numpy as np

# Defining the bits as numpy arrays
zero = np.array([1, 0])  # |0>
one = np.array([0, 1])  # |1>

# Defining some single qubit states
plus = (zero + one) / np.sqrt(2)  # |+> state
minus = (zero - one) / np.sqrt(2)  # |-> state
plusi = (zero + 1j * one) / np.sqrt(2)  # |+i> state
minusi = (zero - 1j * one) / np.sqrt(2)  # |->i> state

# Definind some single qubit gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
X = np.array([[0, 1], [1, 0]])  # Pauli-X gate
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z gate
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T gate
S = np.array([[1, 0], [0, 1j]])  # S gate
I = np.array([[1, 0], [0, 1]])  # Identity gate

# So running a circuit with H, T, X with a qubit initialized to |0> will result in:
# print(X @ T @ H @ zero)


#%% 2 qubit 
zerozero = np.kron(zero, zero)  # |00>
zerone = np.kron(zero, one)  # |01>
onezero = np.kron(one, zero)  # |10>
oneone = np.kron(one, one)  # |11>

# Bell states
bell00 = (np.kron(zero, zero) + np.kron(one, one)) / np.sqrt(2)  # |Φ+>
bell01 = (np.kron(zero, one) + np.kron(one, zero)) / np.sqrt(2)  # |Φ->
bell10 = (np.kron(zero, zero) - np.kron(one, one)) / np.sqrt(2)  # |Ψ+>
bell11 = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)  # |Ψ->

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
#print(CNOT @ (np.kron(H @ zero, H @ zero)))

#%% Defninig a measurement function

# Let's say our starting state is |psi> and a post-measurement state is |psi'>
# Then the probability of measuring a state |j> on the ith qubit is given by:
# P(j) = |||j>_i <j|_i |psi>||^2
# So the post-measurement state is given by:
# |psi'> = |j>_i <j|_i |psi> / || |j>_i <j|_i |psi> ||
# where |j>_0 = |j> x I x ... (identity on all other qubits)

projectors=[np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ] # list containing the projectors |0><0| and |1><1|, for a z-basis projection

def project(i,j,reg): # RETURN state with ith qubit of reg projected onto |j>
    projected=np.tensordot(projectors[j],reg.psi,(1,i))
    return np.moveaxis(projected,0,i)

from scipy.linalg import norm

def measure(i,reg): # Change reg.psi into post-measurement state w/ correct probability. Return measurement value as int
    projected=project(i,0,reg) 
    norm_projected=norm(projected.flatten()) 
    if np.random.random()<norm_projected**2: # Sample according to probability distribution
        reg.psi=projected/norm_projected
        return 0
    else:
        projected=project(i,1,reg)
        reg.psi=projected/norm(projected)
        return 1