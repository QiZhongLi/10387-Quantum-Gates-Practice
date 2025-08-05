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

def measure_qubit(state, qubit_position, basis='Z'):
    """
    Measures a single qubit in a multi-qubit state.
    Args:
        state: The state vector (e.g., |00⟩ = [1, 0, 0, 0]).
        qubit_position: Index of the qubit to measure (0-based).
        basis: 'Z' (computational) or 'X' (Hadamard).
    Returns:
        outcome (0 or 1), collapsed_state
    """
    n_qubits = int(np.log2(len(state)))
    
    # Switch to X-basis if needed
    if basis == 'X':
        H_total = 1
        for i in range(n_qubits):
            H_total = np.kron(H_total, H if i == qubit_position else I)
        state = H_total @ state
    
    # Projectors |0⟩⟨0| and |1⟩⟨1| for the target qubit
    P0 = np.kron(np.eye(2**qubit_position), zero.reshape(-1, 1) @ zero.reshape(1, -1))
    P0 = np.kron(P0, np.eye(2**(n_qubits - qubit_position - 1)))
    
    P1 = np.kron(np.eye(2**qubit_position), one.reshape(-1, 1) @ one.reshape(1, -1))
    P1 = np.kron(P1, np.eye(2**(n_qubits - qubit_position - 1)))
    
    # Probabilities
    prob0 = (state.conj().T @ P0 @ state).real
    prob1 = (state.conj().T @ P1 @ state).real
    
    # Normalize (in case of floating-point errors)
    prob_sum = prob0 + prob1
    prob0, prob1 = prob0 / prob_sum, prob1 / prob_sum
    
    # Random outcome
    outcome = np.random.choice([0, 1], p=[prob0, prob1])
    
    # Collapse the state
    collapsed_state = (P0 if outcome == 0 else P1) @ state
    collapsed_state /= np.linalg.norm(collapsed_state)  # Normalize
    
    return outcome, collapsed_state

#%% Teleportation circuit
alice_state = plusi  # State to teleport (|+⟩)
state = np.kron(alice_state, bell00)  # Combined state (|+⟩ ⊗ |Φ⁺⟩)

# Step 1: Alice applies CNOT and H
state = np.kron(np.kron(H, I), I) @ np.kron(CNOT,I) @ state  # (Simplified; adjust indices as needed)

# Step 2: Measure Alice's qubits
m1, state = measure_qubit(state, qubit_position=0)
m2, state = measure_qubit(state, qubit_position=1)

# Step 3: Bob corrects his qubit (position 2)
if m1 == 1:
    state = np.kron(np.kron(I, I), Z) @ state
    print("Bob applies Z gate")
if m2 == 1:
    state = np.kron(np.kron(I, I), X) @ state
    print("Bob applies X gate")

# Verify Bob's state matches Alice's original |+⟩
#bob_state = state.reshape(2, 2)[:, 0]  # Marginalize Alice's qubits
#print("Bob's state:", bob_state)

print(state)

# %%
