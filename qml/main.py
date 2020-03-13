import os
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from config import get_config, print_usage, print_config
from quantum_network import QuantumNet
from classic_model import ClassicModel


def data_criterion(config):
    """Returns the loss object based on the commandline argument for the data term"""
    return getattr(nn, config.loss_type)()


def model_criterion(config):
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param ** 2)

        return loss * config.l2_reg

    return model_loss


def get_model(config):
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    if config.model == "classic":
        num_ftrs = model_conv.fc.in_features
        # model_conv.fc = nn.Linear(num_ftrs, 2)
        model_conv.fc = ClassicModel(config, num_ftrs)

    elif config.model == "quantum":
        model_conv.fc = QuantumNet(config)

    print(model_conv)

    return model_conv


def train_model(config):
    since = time.time()

    writers = {
        x: SummaryWriter(log_dir=os.path.join(config.log_dir, x)) for x in ['train', 'val']
    }
    iter_idx = -1

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)

    criterion = data_criterion(config)
    # optimizer = optim.SGD(model.fc.parameters(), lr=config.learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.fc.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(config.num_epoch):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                    loss = criterion(outputs, labels)

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
                    "best_va_acc": best_acc
                }, bestmodel_file)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, config):
    """Test routine"""

    data_transforms = {
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(config.crop_size),
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
    # model = get_model(config)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # load_res = torch.load(os.path.join(config.save_dir, "best_model.pth"))
    # model.load_state_dict(load_res["model"])
    # model.eval()

    criterion = data_criterion(config)
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
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_sizes["test"]
    test_acc = running_corrects.double() / dataset_sizes["test"]

    print("Test Loss = {}".format(test_loss))
    print("Test Accuracy = {}%".format(test_acc))


def main(config):
    """The main function."""
    model = train_model(config)
    test(model, config)
    # if config.mode == "train":
    #     train_model(config)
    # elif config.mode == "test":
    #     test(config)
    # else:
    #     raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    print_config(config)
    import time

    main(config)
