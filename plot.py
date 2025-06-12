from utils import *
import pennylane as qml
from pennylane import X, Z, Y, I, compile
# as pennylane has cudaq np implementation
from pennylane import numpy as np
from scipy.linalg import expm
from typing import Union
from pennylane import qchem
# for summing two dictionaries
from collections import defaultdict
from matplotlib import pyplot as plt
from pennylane.transforms import decompose
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TrotterQRTE, TimeEvolutionProblem
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import Statevector
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator

# Defination: params, symbols ,coordinates, molecule, Ham, qubits
symbols = ['H', 'H']
coordinates = np.array([[0.,  0.66140414,0.], [-0.66140414, 0., 0.66140414]])
# taken from https://pennylane.ai/qml/demos/tutorial_mol_geo_opt
# symbols = ["H", "H", "H"]
# coordinates = np.array([[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]])
molecule = qml.qchem.Molecule(symbols, coordinates, charge=0)  # charge=1 for H3+
Ham, qubits = qml.qchem.molecular_hamiltonian(molecule)
coeffs = np.array(Ham.terms()[0])
ops = Ham.terms()[1]
ops[0] = I(0)
H = qml.Hamiltonian(coeffs, ops)
H_matrix = qml.matrix(H)
H_qiskit = SparsePauliOp.from_operator(H_matrix)
# defining some params
eps_arr = [1/2**i for i in range(-3,3)]  # epsilon values
time = 8.0
# for number of iterations for plots
N_iters = 5
target_gate_set = ['RX', 'RY', 'RZ', 'CNOT', 'PauliX', 'PauliY', 'PauliZ', 'S', 'T']

if __name__ == "__main__":
    # iterating over eps
    # having array of fidelity, diamond norm distance, resources...to plot
    diamond_norm_distances_qdrift_arr = []
    diamond_norm_distances_qdrift_alias_arr = []
    diamond_norm_distances_trotter_arr = []
    fidelity_qdrift_arr = []
    fidelity_qdrift_alias_arr = []
    fidelity_trotter_arr = []
    resources_qdrift_arr = []
    resources_qdrift_alias_arr = []
    resources_trotter_arr = []
    resources_qdrift_gate_sizes_arr = []
    resources_qdrift_alias_gate_sizes_arr = []
    resources_trotter_gate_sizes_arr = []

    # for faster computation...reusing the initial state
    initial_state = np.ones(2**qubits)
    initial_state *= 1 / np.sqrt(2**qubits)  # normalized state
    U_ideal_matrix = construct_ideal_unitary(H, time, qubits)
    psi_ideal = U_ideal_matrix @ initial_state
    # running the qdrift circuit for each epsilon
    # for plotting
    for eps in eps_arr:
        print(f"Running for epsilon: {eps}")
        # generate plots of average number of terms vs epsilon for 100 iterations
        # need average of values, storing resources, diamond norm distance, fidelity
        resources_qdrift = {"depth": 0, "num_gates": 0}
        # having different attribute for gate sizes in loop
        resources_qdrift_alias = {"depth": 0, "num_gates": 0}
        resources_trotter = {"depth": 0, "num_gates": 0}

        # for diamond norm distance and fidelity
        diamond_norm_distances_qdrift = 0
        diamond_norm_distances_qdrift_alias = 0
        diamond_norm_distances_trotter = 0
        fidelity_qdrift = 0
        fidelity_qdrift_alias = 0
        fidelity_trotter = 0
        # run the qdrift circuit for N_iters iterations
        # and store the resources, diamond norm distance, fidelity
        for i in range(N_iters):
            print(f"Iteration {i+1}/{N_iters}")
            circuit_normal = qdrift_circuit(H, time=time, epsilon=eps, wires=[i for i in range(qubits)])
            circuit_alias = qdrift_circuit(H, time=time, epsilon=eps, wires=[i for i in range(qubits)],use_alias_sampler = True)
            # computing number of steps for the trotterization
            n = int(time / eps)

            trotter_step_first_order = PauliEvolutionGate(H_qiskit, time, synthesis=LieTrotter(reps=n))
            # We create an empty circuit
            circuit = QuantumCircuit(H_qiskit.num_qubits)
            circuit.append(trotter_step_first_order, range(H_qiskit.num_qubits))
            circuit = circuit.decompose(reps=2)
            circuit.save_unitary()
            simulator = AerSimulator(method = 'unitary')
            circ = transpile(circuit, simulator)

            # Run and get unitary
            result = simulator.run(circ).result()
            unitary = result.get_unitary(circ)

            #summing the resources
            # saving the resources as temp for each iteration
            # for depth and number of gate
            rs_qdrift = qml.specs(circuit_normal, level=0)()["resources"]
            rs_qdrift_alias = qml.specs(circuit_alias, level=0)()["resources"]
            

            resources_qdrift["depth"]+=rs_qdrift.depth
            resources_qdrift_alias["depth"]+=rs_qdrift_alias.depth
            resources_trotter["depth"]+=circuit.depth()
            resources_qdrift["num_gates"]+=rs_qdrift.num_gates  
            resources_qdrift_alias["num_gates"]+=rs_qdrift_alias.num_gates
            resources_trotter["num_gates"]+=len(circuit)
            # formatting gate sizes for qiskit as it contains single and two qubit gates
            single = len(circuit) - circuit.num_nonlocal_gates()
            two_qubit = circuit.num_nonlocal_gates()
            # for gate sizes
            if i == 0:
                resources_qdrift_gate_sizes=rs_qdrift.gate_sizes
                resources_qdrift_alias_gate_sizes=rs_qdrift_alias.gate_sizes

                resources_trotter_gate_sizes = {1: single, 2: two_qubit}
  
            else:
                resources_qdrift_gate_sizes = sum_dicts(rs_qdrift.gate_sizes, 
                                                        resources_qdrift_gate_sizes)
                resources_qdrift_alias_gate_sizes = sum_dicts(rs_qdrift_alias.gate_sizes,
                                                        resources_qdrift_alias_gate_sizes)
                resources_trotter_gate_sizes = sum_dicts({1: single, 2: two_qubit},
                                                        resources_trotter_gate_sizes)

            # gettin unitary matrix for the qiskit trotter circuit
            U_trotter = unitary

            U_qdrift_matrix = qml.matrix(circuit_normal)
            U_qdrift_alias_matrix = qml.matrix(circuit_alias)
            diamond_norm_distances_qdrift += diamond_norm_unitaries(U_qdrift_matrix(), U_ideal_matrix)
            # print(f"Diamond norm distance Qdrift: {norm_dist_qdrift}")
            diamond_norm_distances_qdrift_alias += diamond_norm_unitaries(U_qdrift_alias_matrix(), U_ideal_matrix)
            # print(f"Diamond norm distance Qdrift alias: {norm_dist_qdrift_alias}")

            # compute the fidelity
            psi_qdrift_alias = U_qdrift_alias_matrix() @ initial_state
            fidelity_qdrift_alias += np.abs(np.vdot(psi_qdrift_alias, psi_ideal))**2
            # print(f"Fidelity: {fidelity}")
            # compute diamond norm distance between trotter and ideal
            diamond_norm_distances_trotter  += diamond_norm_unitaries(U_trotter, U_ideal_matrix)
            # print(f"Diamond norm distance Trotter: {norm_dist_trotter}")
            psi_qdrift = U_qdrift_matrix() @ initial_state
            fidelity_qdrift += np.abs(np.vdot(psi_qdrift, psi_ideal))**2
            psi_trotter = U_trotter @ initial_state
            fidelity_trotter += np.abs(np.vdot(psi_trotter, psi_ideal))**2

        # average the resources, diamond norm distance, fidelity
        resources_qdrift = {k: v / N_iters for k, v in resources_qdrift.items()}
        resources_qdrift_gate_sizes = {k: v / N_iters for k, v in resources_qdrift_gate_sizes.items()}
        resources_qdrift_alias = {k: v / N_iters for k, v in resources_qdrift_alias.items()}
        resources_qdrift_alias_gate_sizes = {k: v / N_iters for k, v in resources_qdrift_alias_gate_sizes.items()}
        resources_trotter = {k: v / N_iters for k, v in resources_trotter.items()}  
        resources_trotter_gate_sizes = {k: v / N_iters for k, v in resources_trotter_gate_sizes.items()}
        diamond_norm_distances_qdrift /= N_iters
        diamond_norm_distances_qdrift_alias /= N_iters
        diamond_norm_distances_trotter /= N_iters
        fidelity_qdrift /= N_iters
        fidelity_qdrift_alias /= N_iters
        fidelity_trotter /= N_iters

        # print the average resources
        print("Average resources for Qdrift:", resources_qdrift)
        print("Average gate sizes for Qdrift:", resources_qdrift_gate_sizes)
        print("Average resources for Qdrift alias:", resources_qdrift_alias)
        print("Average gate sizes for Qdrift alias:", resources_qdrift_alias_gate_sizes)
        print("Average resources for Trotter:", resources_trotter)
        print("Average gate sizes for Trotter:", resources_trotter_gate_sizes)

        # print the average diamond norm distances and fidelity
        print("Average diamond norm distance for Qdrift:", diamond_norm_distances_qdrift)
        print("Average diamond norm distance for Qdrift alias:", diamond_norm_distances_qdrift_alias)
        print("Average diamond norm distance for Trotter:", diamond_norm_distances_trotter)
        print("Average fidelity for Qdrift:", fidelity_qdrift)
        print("Average fidelity for Qdrift alias:", fidelity_qdrift_alias)
        print("Average fidelity for Trotter:", fidelity_trotter)

        # append the results to the arrays for plotting
        diamond_norm_distances_qdrift_arr.append(diamond_norm_distances_qdrift)
        diamond_norm_distances_qdrift_alias_arr.append(diamond_norm_distances_qdrift_alias)
        diamond_norm_distances_trotter_arr.append(diamond_norm_distances_trotter)
        fidelity_qdrift_arr.append(fidelity_qdrift)
        fidelity_qdrift_alias_arr.append(fidelity_qdrift_alias)
        fidelity_trotter_arr.append(fidelity_trotter)
        resources_qdrift_arr.append(resources_qdrift)
        resources_qdrift_alias_arr.append(resources_qdrift_alias)
        resources_trotter_arr.append(resources_trotter)
        resources_qdrift_gate_sizes_arr.append(resources_qdrift_gate_sizes)
        resources_qdrift_alias_gate_sizes_arr.append(resources_qdrift_alias_gate_sizes)
        resources_trotter_gate_sizes_arr.append(resources_trotter_gate_sizes)
    # plotting the results as individual plots
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(eps_arr, diamond_norm_distances_qdrift_arr, label='Qdrift', marker='o')
    plt.plot(eps_arr, diamond_norm_distances_qdrift_alias_arr, label='Qdrift Alias', marker='o')
    plt.plot(eps_arr, diamond_norm_distances_trotter_arr, label='Trotter', marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Diamond Norm Distance')         
    plt.title('Diamond Norm Distance vs Epsilon')
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 2)
    plt.plot(eps_arr, fidelity_qdrift_arr, label='Qdrift', marker='o')
    plt.plot(eps_arr, fidelity_qdrift_alias_arr, label='Qdrift Alias', marker='o')
    plt.plot(eps_arr, fidelity_trotter_arr, label='Trotter', marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Epsilon')
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 3)
    plt.plot(eps_arr, [r['depth'] for r in resources_qdrift_arr], label='Qdrift', marker='o')
    plt.plot(eps_arr, [r['depth'] for r in resources_qdrift_alias_arr], label='Qdrift Alias', marker='o')
    plt.plot(eps_arr, [r['depth'] for r in resources_trotter_arr], label='Trotter', marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Average Circuit Depth')
    plt.title('Average Circuit Depth vs Epsilon')
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 4)
    plt.plot(eps_arr, [r['num_gates'] for r in resources_qdrift_arr], label='Qdrift', marker='o')
    plt.plot(eps_arr, [r['num_gates'] for r in resources_qdrift_alias_arr], label='Qdrift Alias', marker='o')
    plt.plot(eps_arr, [r['num_gates'] for r in resources_trotter_arr], label='Trotter', marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Average Number of Gates')
    plt.title('Average Number of Gates vs Epsilon')
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 5)
    plt.bar(resources_qdrift_gate_sizes_arr[0].keys(), 
            [v for v in resources_qdrift_gate_sizes_arr[0].values()], 
            label='Qdrift', alpha=0.5)
    plt.bar(resources_qdrift_alias_gate_sizes_arr[0].keys(),
            [v for v in resources_qdrift_alias_gate_sizes_arr[0].values()], 
            label='Qdrift Alias', alpha=0.5)
    plt.bar(resources_trotter_gate_sizes_arr[0].keys(), 
            [v for v in resources_trotter_gate_sizes_arr[0].values()], 
            label='Trotter', alpha=0.5)
    plt.xlabel('Gate Sizes')
    plt.ylabel('Average Number of Gates')
    plt.title('Average Number of Gates by Size')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.suptitle('Comparison of Qdrift, Qdrift Alias, and Trotterization for H2', fontsize=16)
    plt.subplots_adjust(top=0.9)  
    plt.savefig('comparison_qdrift_trotter_h2.png')
    plt.show()
