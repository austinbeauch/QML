"""
Circuits from https://arxiv.org/pdf/1905.10876.pdf
"""

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
        ry_layer(w)


class Circuit1(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:])


class Circuit2(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:])
        for i in range(n_qubits):
            qml.CNOT(wires=[i, i + 1])


class Circuit3(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits + qubits - 1
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:-n_qubits + 1])
        for i in range(n_qubits - 1):
            qml.CRZ(w[2 * n_qubits + i], wires=[i, i + 1])


class Circuit4(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits + qubits - 1
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:-n_qubits + 1])
        for i in range(n_qubits - 1):
            qml.CRX(w[2 * n_qubits + i], wires=[i, i + 1])


class Circuit5(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 4 * qubits + qubits * (qubits - 1)
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:2 * n_qubits])

        for i in range(n_qubits):
            for j in range(n_qubits):
                if i == j:
                    continue
                qml.CRZ(2 * n_qubits + i * (n_qubits - 1) + j, wires=[i, j])

        rx_layer(w[-2 * n_qubits:-n_qubits])
        rz_layer(w[-n_qubits:])


class Circuit6(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 4 * qubits + qubits * (qubits - 1)
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:2 * n_qubits])

        for i in range(n_qubits):
            for j in range(n_qubits):
                if i == j:
                    continue
                qml.CRX(2 * n_qubits + i * (n_qubits - 1) + j, wires=[i, j])

        rx_layer(w[-2 * n_qubits:-n_qubits])
        rz_layer(w[-n_qubits:])


class Circuit9(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        h_layer(n_qubits)
        for i in range(n_qubits):
            qml.CZ(wires=[i, i + 1])
        rx_layer(w)


class Circuit11(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        assert qubits % 2 == 0  # unsure if qubits needs to be even to scale properly
        params = 2 * qubits + 2 * qubits//2
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        ry_layer(w[:n_qubits])
        rz_layer(w[n_qubits:2 * n_qubits])
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n_qubits - 1):
            qml.RY(w[2 * n_qubits + i - 1], wires=i)
        for i in range(1, n_qubits - 1):
            qml.RZ(w[2 * n_qubits + n_qubits // 2 + i - 1], wires=i)
        for i in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])


class Circuit19(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 3 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(n_qubits, w):
        rx_layer(w[:n_qubits])
        rz_layer(w[n_qubits:])
        qml.CRX(w[8], wires=[0, 3])
        qml.CRX(w[9], wires=[3, 2])
        qml.CRX(w[10], wires=[2, 1])
        qml.CRX(w[11], wires=[1, 0])
