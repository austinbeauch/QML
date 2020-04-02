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
    def layer(qubits, w):
        entangling_layer(qubits)
        ry_layer(w)


class Circuit1(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:])


class Circuit2(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:])
        for i in range(qubits):
            qml.CNOT(wires=[i, i + 1])


class Circuit3(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits + qubits - 1
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:-qubits + 1])
        for i in range(qubits - 1):
            qml.CRZ(w[2 * qubits + i], wires=[i, i + 1])


class Circuit4(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits + qubits - 1
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:-qubits + 1])
        for i in range(qubits - 1):
            qml.CRX(w[2 * qubits + i], wires=[i, i + 1])


class Circuit5(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 4 * qubits + qubits * (qubits - 1)
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:2 * qubits])

        for i in range(qubits):
            for j in range(qubits):
                if i == j:
                    continue
                qml.CRZ(2 * qubits + i * (qubits - 1) + j, wires=[i, j])

        rx_layer(w[-2 * qubits:-qubits])
        rz_layer(w[-qubits:])


class Circuit6(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 4 * qubits + qubits * (qubits - 1)
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:2 * qubits])

        for i in range(qubits):
            for j in range(qubits):
                if i == j:
                    continue
                qml.CRX(2 * qubits + i * (qubits - 1) + j, wires=[i, j])

        rx_layer(w[-2 * qubits:-qubits])
        rz_layer(w[-qubits:])


class Circuit9(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        h_layer(qubits)
        for i in range(qubits):
            qml.CZ(wires=[i, i + 1])
        rx_layer(w)


class Circuit11(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        assert qubits % 2 == 0  # unsure if qubits needs to be even to scale properly
        params = 2 * qubits + 2 * qubits // 2
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        ry_layer(w[:qubits])
        rz_layer(w[qubits:2 * qubits])
        for i in range(0, qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, qubits - 1):
            qml.RY(w[2 * qubits + i - 1], wires=i)
        for i in range(1, qubits - 1):
            qml.RZ(w[2 * qubits + qubits // 2 + i - 1], wires=i)
        for i in range(1, qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])


class Circuit13(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 2 * qubits + 2 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        ry_layer(w[:qubits])
        wire_list = list(range(qubits))
        for i in range(qubits):
            qml.CRZ(w[qubits+i], wires=[i, wire_list[i-1]])

        ry_layer(w[2*qubits:3*qubits])

        # this doesn't generate it exactly as outlined in the paper
        iterations = list(range(qubits - 1, -1, -1))
        iterations.append(iterations.pop(0))
        for i in iterations:
            qml.CRZ(w[3*qubits+i], wires=[wire_list[i-1], i])


class Circuit19(QuantumCircuit):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__(qubits, depth, delta, dev)
        params = 3 * qubits
        self.params_per_layer = torch.Tensor([params]).type(torch.int32)
        self.q_params = nn.Parameter(delta * torch.randn(self.params_per_layer * depth))

    @staticmethod
    def layer(qubits, w):
        rx_layer(w[:qubits])
        rz_layer(w[qubits:])
        qml.CRX(w[8], wires=[0, 3])
        qml.CRX(w[9], wires=[3, 2])
        qml.CRX(w[10], wires=[2, 1])
        qml.CRX(w[11], wires=[1, 0])
