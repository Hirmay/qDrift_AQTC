import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import expm
from typing import Union
from pennylane import qchem
from collections import defaultdict
from pennylane import ApproxTimeEvolution
from functools import partial
from pennylane.transforms import decompose

allowed_gates = ['RX', 'RY', 'RZ', 'CNOT', 'PauliX', 'PauliY', 'PauliZ', 'S', 'T']
qml.decomposition.disable_graph()


def sum_dicts(dd1: defaultdict, dd2: defaultdict) -> defaultdict:
    """
    Sums two dicts with integer values, in our case gate_sizes.
    Arguments:
        dd1: defaultdict
            First defaultdict to sum.
        dd2: defaultdict
            Second defaultdict to sum.
    Returns:
        defaultdict for gate_sizes
            Dict containing the sum of the two input defaultdicts.
    """
    result = defaultdict(int)  
    for key in dd1:
        result[key] = dd1[key] + dd2.get(key, 0)  
    return result

def qdrift_generator(hamiltonian:qml.Hamiltonian, time: float, epsilon: float, use_alias_sampler: bool = False)-> Union[list[float] , list[qml.matrix]]:
    """
    The function takes in a Hamiltonian using PennyLane function, total evolution time, and error threshold.
    Computes the sampling probabilities and number of random sequence of terms.
    Arguments:
        hamitonian: qml.Hamiltonian
            user defined Hamiltonian needed for time evolution
        time: float
            total evolution time for the system
        epsilon: float
            error threshold, with incorporating diamond error
    """

    # extracting coefficients and operators
    coeffs = np.abs(np.array(hamiltonian.coeffs))
    operators = hamiltonian.ops

    # total weight given by lambda
    lam = np.sum(np.abs(coeffs))

    # no of steps
    N_steps = int(np.ceil((2 * lam*time)**2 / epsilon))

    # probabilities comp
    probabilities = coeffs/lam

    if use_alias_sampler:
        sampler = AliasSampler(probabilities)
        chosen_indices = list(sampler.sample(size=N_steps))
    else:
        # np.random.choice requires p to sum to 1.
        chosen_indices = list(np.random.choice(len(operators), size=N_steps, p=probabilities))

    # Precomputing the respective unitaries for each term
    unitary_factors = []
    for j, operator in enumerate(operators):
        # matrix representation of the term
        matrix = qml.matrix(operator)
        # sign of original coeffs
        h_j = hamiltonian.coeffs[j]
        # scaled generator
        theta = np.sign(h_j) * lam * time/ N_steps
        U = expm (-1j * theta * matrix)
        unitary_factors.append({'matrix': U, 'wires': operator.wires})
    return chosen_indices, unitary_factors

def construct_ideal_unitary(hamiltonian: qml.Hamiltonian, time: float, num_qubits: int) -> np.ndarray:
    """Constructs the ideal unitary matrix U_ideal = exp(-iHt)."""
    H_matrix = qml.matrix(hamiltonian, wire_order=range(num_qubits)) 
    U_ideal = expm(-1j * H_matrix * time)
    return U_ideal

def qdrift_circuit(hamiltonian: qml.Hamiltonian, time: float, epsilon: float, wires: qml.wires, use_alias_sampler: bool = False) -> qml.QNode :
    """
    Returns a Pennylane QNode approximating exp(-i H t) via Qdrift giving the final state. 
    Arguments:
        hamitonian: qml.Hamiltonian
            user defined Hamiltonian needed for time evolution
        time: float
            total evolution time for the system
        epsilon: float
            error threshold
    
    """
    dev = qml.device("default.qubit", wires = wires)
    indices_chosen, unitaries = qdrift_generator(hamiltonian, time = time, epsilon=epsilon, use_alias_sampler=use_alias_sampler)

    @qml.qnode(dev)
    def circuit():
        # init state (can customise)
        for idx in indices_chosen:
            op_data = unitaries[idx]
            U = op_data['matrix']
            op_wires = op_data['wires']
            qml.QubitUnitary(U, wires=op_wires)
            # apply unitaries on corresponding wires
        # return the final state

        return qml.state()
        # return qml.matrix(wires=wires)
    return circuit

# Unitary matrix for the first channel (in our case qDrift)
# other being ideal
def diamond_norm_unitaries(U_1: np.ndarray, U_2: np.ndarray) -> float:

    """
    Calculates the diamond norm distance between two unitary channels.
    Formula: ||E_U1 - E_U2||_diamond = 1/2 * sqrt(1 - (|Tr(U1^dagger U2)|/d)^2)
    where d is the dimension of the Hilbert space (2^num_qubits).
    """
    dim = U_1.shape[0]
    trace_val = np.trace(np.conjugate(U_1.T) @ U_2)
    abs_trace_val = np.abs(trace_val)
    
    # The term inside the square root: 1 - (|Tr(V_dagger U)|/d)^2
    term_sqrt = 1 - (abs_trace_val / dim)**2

    diamond_dist = 0.5* np.sqrt(term_sqrt)
    return diamond_dist

class AliasSampler:
    """
    Implementing author's alias method for efficient sampling from a discrete distribution
    based on the probabilities provided.
    It uses the alias method for efficient sampling.
    Arguments:
        probs: list[float]
            A list of probabilities that sum to 1.0, representing the discrete distribution.
    """
    
    def __init__(self, probs):
        self.n = len(probs)
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=int)
        self._init_alias(probs)

    def _init_alias(self, probs):
        scaled = np.array(probs) * self.n
        small = [i for i, v in enumerate(scaled) if v < 1.0]
        large = [i for i, v in enumerate(scaled) if v >= 1.0]

        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s] = scaled[s]
            self.alias[s] = l
            scaled[l] = scaled[l] - (1.0 - scaled[s])
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)

        for leftover in large + small:
            self.prob[leftover] = 1.0

    def sample(self, size=1):
        idx = np.random.randint(0, self.n, size)
        flips = np.less(np.random.rand(size), self.prob[idx])
        return np.where(flips, idx, self.alias[idx])