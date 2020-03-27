import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as np


class DefaultCircuit(nn.Module):
    """
    Custom variational quantum circuit Torch module.
    """

    def __init__(self, qubits, depth, delta, dev):
        super().__init__()
        self.qubits = torch.Tensor([qubits]).type(torch.int32)
        self.depth = torch.Tensor([depth]).type(torch.int32)
        self.delta = delta

        self.q_params = nn.Parameter(delta * torch.randn(depth * qubits))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnode = qml.QNode(_q_net, dev, interface="torch")

    def forward(self, q_in):
        # Apply the quantum circuit to each element of the batch
        q_out = torch.Tensor(0, self.qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = self.qnode(elem, self.q_params, self.depth, self.qubits).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        return q_out

    def extra_repr(self):
        return 'qubits={}, depth={}, delta={}'.format(
            self.qubits.item(), self.depth.item(), self.delta
        )


def _q_net(q_in, q_weights_flat, q_depth, n_qubits):
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

