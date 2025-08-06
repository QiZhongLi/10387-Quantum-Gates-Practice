import numpy as np
from scipy.linalg import norm 


# Playing with Joris Kattemölle's code and see how his stuff works
H_matrix=1/np.sqrt(2)*np.array([[1, 1],
                                [1,-1]])

X = np.array([[0, 1], [1, 0]])  # Pauli-X gate
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z gate
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T gate
S = np.array([[1, 0], [0, 1j]])  # S gate
I = np.array([[1, 0], [0, 1]])  # Identity gate

CNOT_matrix=np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1],
                      [0,0,1,0]])

CNOT_tensor=np.reshape(CNOT_matrix, (2,2,2,2))

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
    reg.psi=np.moveaxis(reg.psi,0,i)

def X(i,reg):
    reg.psi=np.tensordot(X,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Z(i,reg):
    reg.psi=np.tensordot(Z,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Y(i,reg):
    reg.psi=np.tensordot(Y,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def CNOT(control, target, reg):
    reg.psi=np.tensordot(CNOT_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))   

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
    
# Example of final usage: create uniform superposition
reg=Reg(3)


'''
H(2, reg)  # Alice qubit

#initializing bell state
H(1, reg)
CNOT(1,0, reg)

# Teleportation protocol
CNOT(2,1, reg)
H(2, reg)
m1 = measure(2, reg)
m2 = measure(1, reg)

if m1 == 1:
    Z(0, reg)
if m2 == 1:
    X(0, reg)

# Final state
state = reg.psi.flatten()
print("Final state after teleportation:", state)'''