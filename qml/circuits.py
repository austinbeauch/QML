import torch
from torch import nn
import pennylane as qml

from layers import *
from quantum_circuit import QuantumCircuit


class XanaduCircuit(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(params * depth))

    @staticmethod
    def layer(n_qubits, w):
        entangling_layer(n_qubits)
        RY_layer(w)


class Circuit1(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        RX_layer(w[:n_qubits])
        RZ_layer(w[n_qubits:])


class Circuit19(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 3 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        RX_layer(w[:n_qubits])
        RZ_layer(w[n_qubits:])
        qml.CRX(w[8], wires=[0, 3])
        qml.CRX(w[9], wires=[3, 2])
        qml.CRX(w[10], wires=[2, 1])
        qml.CRX(w[11], wires=[1, 0])
