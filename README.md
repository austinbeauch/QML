# QML
Quantum Machine Learning Undergraduate Project Repo

This work is based off the [Xanadu AI tutorial](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html]) [1]
 for quantum transfer learning and aims to provide a framework for quantum transfer learning applications. 
 The Xanadu notebook as been refactored into a more traditional machine learning library, consisting of a main 
 testing/training script, a classical neural network module, and various quantum modules. The purpose is to give 
 researchers an library for developing and testing quantum machine learning algorithms and architectures.
 Currently, only a single quantum variational circuit has been implemented with tuneable hyperparameters from
 the command line.
 

## Usage

After cloning and installing all the requirements from qml/requirements.txt, the main training/testing function
is called via

```shell script
~/QML/qml$ python main.py 
```

To see a list of optional parameters for switching modes between quantum models and classical models and tuning 
hyperparamteres:

```shell script
~/QML/qml$ python main.py -h
```

Example training usage might be:

```shell script
~/QML/qml$ python main.py --model quantum --num_epoch 5 --depth 5 --n_qubits 4 --mode train
```

(note that n_qubits refers to the number of nodes in each layer of a classical analogue. Interesting for comparing
quantum to classical for the same architecture, however on our hymenoptera dataset the default depth of 6 of 4 nodes
results in overfiting)

## Creating Custom Circuits
To create and test custom quantum variational circuits, see [citcuits.py](https://github.com/austinbeauch/QML/blob/master/qml/circuits.py).
There are already some custom circuits which have been implemented from [2]. A custom circuit class must be derived from 
the class `QuantumCircuit`, which defines the circuits device and forward pass. The only thing required from a custom
circuit class is the torch parameter definitions, which can be seen in the `__init__` function below. Then, define
each layer of the circuit in the `layer` method. `rx_layer` and `ry_layer` are from the `layers.py` module, which
can be expanded for ease of use. The inputs into `layer` are `qubits`, which defines the number of qubits used for the
circuit, and `w`, which are the weights for any tunable gates.

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

### Calling Custom Circuits
Calling a custom circuit class can be done with the `--circuit` command line argument, where the following string must 
match the class name from `circuits.py` (**case sensitive**). For example:

```shell script
~/QML/qml$ python main.py --num_epoch 10 --depth 2 --n_qubits 4 --circuit Circuit2
```

### References 

[1] A. Mari, T. R. Bromley, J. Izaac, M. Schuld, and N. Killo-ran, “Transfer learning in hybrid classical-quantum neu-ral networks,”arXiv:1912.08278 [quant-ph, stat], Dec.2019.

[2] S. Sim, et al. “Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum‐Classical Algorithms.” (2019).

