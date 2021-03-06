"""
Quantum transfer learning framework main script. This work is based off the Xanadu AI tutorial from the link below.

https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html
"""

import os
import copy
import shutil
import time

import torch
import numpy as np
import pennylane as qml
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

from circuits import *
from config import get_config, print_usage, print_config
from quantum_network import QuantumNet
from classic_model import ClassicModel


def get_model(config):
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    if config.model == "classic":
        num_ftrs = model_conv.fc.in_features
        # model_conv.fc = nn.Linear(num_ftrs, 2)
        model_conv.fc = ClassicModel(config, num_ftrs)

    elif config.model == "quantum":

        if config.backend is not None:
            IBMQ.load_account()
            dev = qml.device('qiskit.ibmq', wires=config.qubits, backend=config.backend)

        elif config.dev == "real":
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= config.qubits
                                                                     and not x.configuration().simulator
                                                                     and x.status().operational))
            dev = qml.device('qiskit.ibmq', wires=config.qubits, backend=backend.name())

        elif config.dev == "forest.qvm":
            dev = qml.device(config.dev, device=f"{config.qubits}q-qvm")

        elif config.dev == "forest.pyqvm":
            dev = qml.device(config.dev, device=f"{config.qubits}q-pyqvm")

        else:
            dev = qml.device(config.dev, wires=config.qubits)

        print("Running on", dev.name)
        model_conv.fc = QuantumNet(config, dev)

    print(model_conv.fc)

    return model_conv


def train_model(config):
    since = time.time()

    writers = {
        x: SummaryWriter(log_dir=os.path.join(config.log_dir, x)) for x in ['train', 'val']
    }

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    bestmodel_file = os.path.join(config.save_dir,
                                  f"{config.circuit}_{config.qubits}_{config.depth}_{config.epochs}_best.pth")
    trainmodel_file = os.path.join(config.save_dir,
                                   f"{config.circuit}_{config.qubits}_{config.depth}_{config.epochs}_train.pth")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../_data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']
                      }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(f"{dataset_sizes['train']} training photos")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)

    cost = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.fc.parameters(), lr=config.learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.fc.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7)

    iter_idx = -1
    best_acc = 0

    epoch = 0
    if os.path.exists(bestmodel_file):
        if config.resume:
            print("Checkpoint found!")
            load_res = torch.load(bestmodel_file)
            iter_idx = load_res["iter_idx"]
            best_acc = load_res["best_acc"]
            epoch = load_res["epoch"] + 1
            model.load_state_dict(load_res["model"])
            optimizer.load_state_dict(load_res["optimizer"])
        else:
            os.remove(bestmodel_file)

    best_model_wts = copy.deepcopy(model.state_dict())
    while epoch < config.epochs:
        # for epoch in range(config.epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            elif epoch % config.val_intv != 0 and epoch-config.epochs != 1:  # validate if it's the last epoch
                continue
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            prefix = "{:6s} Epoch {:3d}: ".format(phase, epoch)
            for data in tqdm(dataloaders[phase], desc=prefix):
                iter_idx += 1
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = cost(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writers[phase].add_scalar("loss", epoch_loss, iter_idx)
            writers[phase].add_scalar("accuracy", epoch_acc, iter_idx)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_idx": iter_idx,
                    "best_acc": best_acc,
                    "epoch": epoch
                }, bestmodel_file)

            if phase == "train":
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_idx": iter_idx,
                    "best_va_acc": best_acc,
                    "epoch": epoch
                }, trainmodel_file)

        epoch += 1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(config):
    """Test routine"""

    data_transforms = {
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_dir = '../_data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model
    model = get_model(config)
    if torch.cuda.is_available():
        model = model.cuda()

    bestmodel_file = os.path.join(config.save_dir,
                                  f"{config.circuit}_{config.qubits}_{config.depth}_{config.epochs}_best.pth")
    trainmodel_file = os.path.join(config.save_dir,
                                   f"{config.circuit}_{config.qubits}_{config.depth}_{config.epochs}_train.pth")

    try:
        print(f"Trying to load {bestmodel_file}...")
        load_res = torch.load(bestmodel_file)
        print("Loaded.")
    except FileNotFoundError:
        print(f"File not found, using {trainmodel_file} instead.")
        load_res = torch.load(trainmodel_file)
    model.load_state_dict(load_res["model"])
    model.eval()

    cost = nn.CrossEntropyLoss()
    prefix = "Testing: "
    running_loss = 0.0
    running_corrects = 0
    for data in tqdm(dataloaders["test"], desc=prefix):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = cost(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_sizes["test"]
    test_acc = running_corrects.double() / dataset_sizes["test"]

    print("Test Loss = {}".format(test_loss))
    print("Test Accuracy = {}%".format(test_acc))

    if config.visualize:
        visualize_model(model, dataloaders, device)


def visualize_model(model, dataloaders, device, num_images=6, fig_name="Predictions"):
    images_so_far = 0
    _fig = plt.figure(fig_name)
    model.eval()
    class_names = {0: "Ants", 1: "Bees"}
    with torch.no_grad():
        for _i, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")

                ax.set_title("Pred: {}, True: {}".format(class_names[preds[j].item()], class_names[labels[j].item()],
                                                         c="g" if preds[j].item() == labels[j].item() else "r"))

                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    plt.show()
                    return


def imshow(inp, title=None):
    """Display image from tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Inverse of the initial normalization operation.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def main(config):
    """The main function."""
    # model = train_model(config)
    # test(model, config)
    if config.mode == "train":
        train_model(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    print_config(config)
    if not config.resume:
        try:
            shutil.rmtree(config.log_dir)
            time.sleep(5)  # sleep to tensorboard doesn't cache old logs
        except FileNotFoundError:
            pass
    main(config)
