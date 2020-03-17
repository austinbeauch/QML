import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as np


# TODO: Find out the QuantumCircuit bottleneck, where is the simulation slow
class QuantumCircuit(nn.Module):
    """
    Custom variational quantum circuit Torch module.
    """

    def __init__(self, qubits, depth, delta):
        super().__init__()
        self.qubits = torch.Tensor([qubits]).type(torch.int32)
        self.depth = torch.Tensor([depth]).type(torch.int32)
        self.delta = delta

        self.q_params = nn.Parameter(delta * torch.randn(depth * qubits))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dev = qml.device("default.qubit", wires=qubits)
        self.q_net = (qml.qnode(dev, interface="torch"))(self._q_net)

    def forward(self, q_in):
        # Apply the quantum circuit to each element of the batch
        q_out = torch.Tensor(0, self.qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = self.q_net(elem, self.q_params, self.depth, self.qubits).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return q_out

    def extra_repr(self):
        return 'qubits={}, depth={}, delta={}'.format(
            self.qubits.item(), self.depth.item(), self.delta
        )

    @staticmethod
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
