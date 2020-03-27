import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as np

from .quantum_circuit import QuantumCircuit
from .layers import *


class XanaduCircuit(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        self.q_params = nn.Parameter(delta * torch.randn(depth * qubits))

    @staticmethod
    def layer(n_qubits, w):
        entangling_layer(n_qubits)
        RY_layer(w)
