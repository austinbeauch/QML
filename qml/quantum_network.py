import importlib

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

from config import get_config
import circuits


class QuantumNet(nn.Module):
    def __init__(self, config, dev):
        super().__init__()
        self.pre_net = nn.Linear(512, config.qubits)

        module = getattr(circuits, f"{config.circuit}")
        self.q_net = module(config.qubits, config.depth, config.q_delta, dev)

        self.post_net = nn.Linear(config.qubits, 2)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = self.q_net(q_in)
        return self.post_net(q_out)
