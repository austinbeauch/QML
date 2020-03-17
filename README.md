# QML
Quantum Machine Learning Undergraduate Project Repo

This work is based off the [Xanadu AI tutorial](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html])
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

# TODO:
* Determine where the quantum network bottlenecks during training
* Add functionality to run on different devices (i.e. use Qiskit to run on real quantum hardware)
* Generate different quantum layers and compare results   


