import torch
from torch import nn
import pennylane as qml

from .layers import *
from .quantum_circuit import QuantumCircuit


class Circuit19(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        self.q_params = nn.Parameter(delta * torch.randn(3 * depth * qubits))

    @staticmethod
    def layer(n_qubits, w):
        RX_layer(w[:4])
        RZ_layer(w[4:8])
        qml.CRX(w[8], wires=[0, 3])
        qml.CRX(w[9], wires=[3, 2])
        qml.CRX(w[10], wires=[2, 1])
        qml.CRX(w[11], wires=[1, 0])
