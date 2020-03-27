import torch
from torch import nn
import pennylane as qml

from .layers import *


class QuantumCircuit(nn.Module):
    def __init__(self, qubits, depth, delta, dev):
        super().__init__()
        self.qubits = torch.Tensor([qubits]).type(torch.int32)
        self.depth = torch.Tensor([depth]).type(torch.int32)
        self.delta = delta
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dev = dev

    def forward(self, q_in):
        q_out = torch.Tensor(0, self.qubits)
        q_out = q_out.to(self.torch_device)
        for elem in q_in:
            q_out_elem = self.quantumcirc(elem,
                                          self.q_params,
                                          self.depth,
                                          self.qubits,
                                          self.params_per_layer).float().unsqueeze(0)

            q_out = torch.cat((q_out, q_out_elem))

        return q_out

    def extra_repr(self):
        return 'qubits={}, depth={}, delta={}'.format(
            self.qubits.item(), self.depth.item(), self.delta
        )

    def quantumcirc(self, *args):
        @qml.qnode(self.dev, interface="torch")
        def q_net(q_in, q_weights_flat, q_depth, n_qubits, params_per_layer):
            q_depth = q_depth[0].val.astype(int)
            n_qubits = n_qubits[0].val.astype(int)
            params_per_layer = params_per_layer[0].val.astype(int)

            q_weights = q_weights_flat.reshape(q_depth, params_per_layer)

            # embedding layers for each circuit
            H_layer(n_qubits)
            RY_layer(q_in)

            # calling the variational layers
            for k in range(q_depth):
                self.layer(n_qubits, q_weights[k])

            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
            return tuple(exp_vals)

        return q_net(*args)

    @staticmethod
    def layer(n_qubits, w):
        raise NotImplementedError("Overwrite me!")
