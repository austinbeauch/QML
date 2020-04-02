"""
Quantum network class.
This work is based off the Xanadu AI tutorial from the link below.

https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html
"""

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
        # for param in self.pre_net.parameters():
        #     param.requires_grad = False

        module = getattr(circuits, f"{config.circuit}")
        self.q_net = module(config.qubits, config.depth, config.q_delta, dev)

        self.post_net = nn.Linear(config.qubits, 2)
        # for param in self.post_net.parameters():
        #     param.requires_grad = False

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = self.q_net(q_in)
        return self.post_net(q_out)
