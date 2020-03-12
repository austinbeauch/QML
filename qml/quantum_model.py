import torch
import pennylane as qml
from pennylane import numpy as np
import torch.nn as nn
from config import get_config


class QuantumNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.pre_net = nn.Linear(512, config.n_qubits)
        self.q_params = nn.Parameter(config.q_delta * torch.randn(config.q_depth * config.n_qubits))
        self.post_net = nn.Linear(config.n_qubits, 2)

        self.n_qubits = torch.Tensor([config.n_qubits]).type(torch.uint8)
        self.q_depth = torch.Tensor([config.q_depth]).type(torch.uint8)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dev = qml.device("default.qubit", wires=4)
        self.q_net = (qml.qnode(dev, interface="torch"))(self._q_net)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = self.q_net(elem, self.q_params, self.q_depth, self.n_qubits).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)

    @staticmethod
    def _q_net(q_in, q_weights_flat, q_depth, n_qubits):
        # Reshape weights
        q_depth = q_depth[0].val.astype(int)
        n_qubits = n_qubits[0].val.astype(int)
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        H_layer(n_qubits)

        # Embed features in the quantum node
        RY_layer(q_in)

        # Sequence of trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
        return tuple(exp_vals)


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
