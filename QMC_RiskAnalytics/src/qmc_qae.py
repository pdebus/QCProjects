from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from qiskit.primitives import StatevectorSampler

import numpy as np

from qmc_distributions import BinomialTreeModel
from risk_measures import ValueAtRisk


class AmplitudeEstimation:

    def __init__(self, num_qae_qubits, qmc_model):

        self.num_qae_qubits = num_qae_qubits
        self.qmc_model = qmc_model

    def qae_circuit(self):

        qmc_circuit = self.qmc_model.risk_measure_circuit()


if __name__ == '__main__':
    pass
