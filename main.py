import numpy as np

#%% 1 qubit stuff

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

#%% 2 qubits stuff
zerozero = np.kron(zero, zero)  # |00>
zeroone = np.kron(zero, one)  # |01>
onezero = np.kron(one, zero)  # |10>
oneone = np.kron(one, one)  # |11>

# Bell states
bell00 = (zerozero + oneone) / np.sqrt(2)  # |Φ+>
bell01 = (zeroone + onezero) / np.sqrt(2)  # |Φ->
bell10 = (zerozero - oneone) / np.sqrt(2)  # |Ψ+>
bell11 = (zeroone - onezero) / np.sqrt(2)  # |Ψ->

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

'''

    # Alice's operations
    # CNOT(Alice's qubit, Alice's Bell qubit)
    state = np.kron(CNOT, np.eye(2)) @ state
    #print("After CNOT:", state) Debugging line
    
    # H on Alice's qubit
    state = np.kron(np.eye(4),H) @ state
    #print("After Hadamard:", state) Debugging line
    
'''

#%% Teleportation circuit

def quantum_teleportation(alice):

    print("Alice's initial state:", alice)
    state = np.kron(alice, bell00)
    
    # Alice's operations
    # CNOT between Alice's state qubit (0) and Alice's Bell qubit (1)
    state = np.kron(CNOT, np.eye(2)) @ state
    # Hadamard on Alice's state qubit (0)
    state = np.kron(H, np.eye(4)) @ state

    #print("State after Alice's operations:", state)
    
    # Measuring the qubits
    # Finding the probabilities of the outcomes
    probabilities = np.abs(state.flatten())**2
    rng = np.random.default_rng() 
    outcome = rng.choice(8, p=probabilities)

    collapsed = np.zeros_like(state)
    collapsed[outcome] = 1.0
    #print("Collapsed state after measurement:", collapsed)

    bits = format(outcome, '03b')  # Convert outcome to 3-bit binary string
    a, b, c = int(bits[0]), int(bits[1]), int(bits[2])
    print("Measurement outcome:", bits, "-> a:", a, "b:", b, "c:", c)


    bob_qubit = None
    if a == 1:
        state = np.kron(np.eye(4), Z) @ state
        print("Bob applied Z gate on his qubit.")
    if b == 1:
        state = np.kron(np.eye(4), X) @ state
        print("Bob applied X gate on his qubit.")
    if a ==0 and b == 0:
        state = state
        print("Bob did not apply any gate on his qubit.")
    
    print("State after Bob's corrections:", state)

    # Reshape state to tensor form 
    state_tensor = state.reshape((2, 2, 2))
    #print(state_tensor)
    # Trace out Alice's qubits by summing over their dimensions
    bob_state = np.sum(state_tensor, axis=(0,1))
    bob_state /= np.linalg.norm(bob_state)  # Normalize
    print("Bob's final state:", bob_state)
 


quantum_teleportation(alice = plus)
