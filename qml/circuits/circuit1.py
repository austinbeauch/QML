import torch
from torch import nn
import pennylane as qml

from .layers import *
from .quantum_circuit import QuantumCircuit


class Circuit1(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        RX_layer(w[:4])
        RZ_layer(w[4:8])
