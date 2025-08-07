import numpy as np
from scipy.linalg import norm 


# Playing with Joris Kattemölle's code and see how his stuff works
# End of the day, we're still working with ndarrays

H_matrix=1/np.sqrt(2)*np.array([[1, 1],
                                [1,-1]])

X_matrix = np.array([[0, 1], [1, 0]])  # Pauli-X gate
Z_matrix = np.array([[1, 0], [0, -1]])  # Pauli-Z gate
Y_matrix = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate
T_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T gate
S_matrix = np.array([[1, 0], [0, 1j]])  # S gate
I_matrix = np.array([[1, 0], [0, 1]])  # Identity gate

CNOT_matrix=np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1],
                      [0,0,1,0]])

CNOT_tensor=np.reshape(CNOT_matrix, (2,2,2,2))

CZ_tensor = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]]).reshape(2, 2, 2, 2)

swap_tensor = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]]).reshape(2, 2, 2, 2)

class Reg: 
    def __init__(self,n):
        #Defininig numebr of qubits we want
        self.n=n
        # In vector form. Initialized to |0> states
        self.psi=np.zeros((2,)*n) 
        self.psi[(0,)*n]=1
        
def H(i,reg): 
    # Applying the Hadamard gate on the i-th qubit
    # So reg.psi would look like [0,1], ..., [1/sqrt(2), 1/sqrt(2)], ...
    reg.psi=np.tensordot(H_matrix,reg.psi,(1,i)) 

    # Reshaping the tensor to move the i-th qubit to the front
    # From what I can test, it doesn't seem to do anything
    # But I'm scare of removing it so I'll keep it for now
    reg.psi=np.moveaxis(reg.psi,0,i)

def X(i,reg):
    reg.psi=np.tensordot(X_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Z(i,reg):
    reg.psi=np.tensordot(Z_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Y(i,reg):
    reg.psi=np.tensordot(Y_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def CNOT(control, target, reg):
    reg.psi=np.tensordot(CNOT_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))   

def CZ(control, target, reg):
    reg.psi = np.tensordot(CZ_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi = np.moveaxis(reg.psi,(0,1),(control,target))

def SWAP(i,j,reg):
    reg.psi = np.tensordot(swap_tensor, reg.psi, ((2, 3), (i, j)))
    reg.psi = np.moveaxis(reg.psi, (0, 1), (i, j))

def CZZ(theta, control, target, reg):
    """
    Apply a controlled rotation around the Z axis.
    theta: angle of rotation
    control: index of the control qubit
    target: index of the target qubit
    """
    RZ_matrix = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, np.exp(1j*theta)]]).reshape(2, 2, 2, 2)
    reg.psi = np.tensordot(RZ_matrix, reg.psi, (1, target))
    reg.psi = np.moveaxis(reg.psi, 0, target)

def measure(i,reg): 
    projectors=[ np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ] 
    
    def project(i,j,reg): 
        projected=np.tensordot(projectors[j],reg.psi,(1,i))
        return np.moveaxis(projected,0,i)
    
    projected=project(i,0,reg) 
    norm_projected=norm(projected.flatten()) 
    if np.random.random()<norm_projected**2: 
        reg.psi=projected/norm_projected
        return 0
    else:
        projected=project(i,1,reg)
        reg.psi=projected/norm(projected)
        return 1

def partial_trace(psi, keep_qubits, n_qubits):
    """
    Trace out all qubits except those in keep_qubits
    psi: state vector (already flattened)
    keep_qubits: list of qubit indices to keep (0-based)
    n_qubits: total number of qubits in original system
    """
    # Reshape to tensor with n_qubits dimensions
    psi_tensor = psi.reshape([2]*n_qubits)
    
    # Sum over all axes NOT in keep_qubits
    trace_axes = [i for i in range(n_qubits) if i not in keep_qubits]
    rho = np.tensordot(psi_tensor, psi_tensor.conj(), axes=(trace_axes, trace_axes))
    
    # Reorder remaining axes if needed
    if len(keep_qubits) > 1:
        new_order = [keep_qubits.index(i) for i in sorted(keep_qubits)]
        rho = np.moveaxis(rho, range(len(keep_qubits)), new_order)
    
    return rho.reshape(2**len(keep_qubits), 2**len(keep_qubits))



#%% Quantum teleportation circuit
'''reg=Reg(3)

# 2 - Alice qubit
# 1 - Alice's Bell qubit
# 0 - Bob's Bell qubit
H(2, reg)  # Alice qubit

#initializing bell state
H(1, reg)
CNOT(1,0, reg)

# Teleportation protocol
CNOT(2,1, reg)
H(2, reg)
m1 = measure(2, reg)
m2 = measure(1, reg)

# Bob applies gates based on outcomes
if m1 == 1:
    Z(0, reg)
if m2 == 1:
    X(0, reg)

# Final state
state = reg.psi.flatten()

# Tracing out other qubits to get Bob's state
bob_density_matrix = partial_trace(state, keep_qubits=[0], n_qubits=3)
bob_state = bob_density_matrix[:, 0]
bob_state /= np.linalg.norm(bob_state)


print("Final state after teleportation:", bob_state)'''

#%% Quantum Fourier Transform (QFT)
reg = Reg(3)