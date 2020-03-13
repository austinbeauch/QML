import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

from config import get_config
from quantum_module import QuantumCircuit


class QuantumNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_net = nn.Linear(512, config.n_qubits)
        self.q_net = QuantumCircuit(config.n_qubits, config.q_depth, config.q_delta)
        self.post_net = nn.Linear(config.n_qubits, 2)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = self.q_net(q_in)
        return self.post_net(q_out)
