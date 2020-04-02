import pennylane as qml


def h_layer(qubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(qubits):
        qml.Hadamard(wires=idx)


def ry_layer(rotations):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, rot in enumerate(rotations):
        qml.RY(rot, wires=idx)


def rx_layer(rotations):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, rot in enumerate(rotations):
        qml.RX(rot, wires=idx)


def rz_layer(rotations):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, rot in enumerate(rotations):
        qml.RZ(rot, wires=idx)


def entangling_layer(qubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, qubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, qubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
