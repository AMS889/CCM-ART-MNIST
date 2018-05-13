import helper
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

rs = 2018
random.seed(rs)

def get_model(n_layers, dropout_rate=0.2, n_filters=10, filter_size=5, fc_units=50):
    """
    Returns a CNN with n_layers number of layers and specified hyperparameters
    """
    if n_layers == 1:
        return Net1(dropout_rate=dropout_rate, n_filters=n_filters, filter_size=filter_size, fc_units=fc_units)
    elif n_layers == 2:
        return Net2(dropout_rate=dropout_rate, n_filters=n_filters, filter_size=filter_size, fc_units=fc_units)
    elif n_layers == 3:
        return Net3(dropout_rate=dropout_rate, n_filters=n_filters, filter_size=filter_size, fc_units=fc_units)
    elif n_layers == 4:
        return Net4(dropout_rate=dropout_rate, n_filters=n_filters, filter_size=filter_size, fc_units=fc_units)
    else:
        raise ValueError('Invalid number of layers. Must be between 1 and 4.')



class Net1(nn.Module) :
    """
    Implements a trainable CNN with one convolutional layer and two fully connected layers
    #### PARAMETERS ####
    dropout_rate: dropout regularization rate
    n_filters: number of filters for to create
    filter_size: size of filters in filter_size x filter_size
    fc_units: numer of units in the first fully-connected layer
    """
    def __init__(self, dropout_rate=0.2, n_filters=10, filter_size=5, fc_units=50) :
        super(Net1, self).__init__()
        self.dropout_rate=dropout_rate
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.fc_units = fc_units
        self.conv1  = nn.Conv2d(1, self.n_filters, kernel_size=self.filter_size, padding=2)
        self.fc1 = nn.Linear(self.n_filters*1*14*14, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, 10)

    def forward(self, x) :
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = x1.view(-1, self.n_filters*14*14)
        x2 = F.relu(self.fc1(x2))
        output = F.dropout(x2, p=self.dropout_rate, training=self.training)
        output = F.log_softmax(self.fc2(output), dim=1)
        return (output, x1, x2)

class Net2(nn.Module) :
    """
    Implements a trainable CNN with four convolutional layers and two fully connected layers
    #### PARAMETERS ####
    dropout_rate: dropout regularization rate
    n_filters: number of filters for to create
    filter_size: size of filters in filter_size x filter_size
    fc_units: numer of units in the first fully-connected layer
    """
    def __init__(self, dropout_rate=0.2, n_filters=10, filter_size=5, fc_units=50) :
        super(Net2, self).__init__()
        self.dropout_rate=dropout_rate
        self.n_filters = n_filters
        self.filter_size = filter_size #
        self.fc_units = fc_units
        self.conv1  = nn.Conv2d(1, self.n_filters, kernel_size=self.filter_size, padding=2)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters*2, kernel_size=self.filter_size, padding=2)
        self.fc1 = nn.Linear(self.n_filters*2*7*7, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, 10)

    def forward(self, x) :
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = x2.view(-1, self.n_filters*2*7*7)
        x3 = F.relu(self.fc1(x3))
        output = F.dropout(x3, p=self.dropout_rate, training=self.training)
        output = F.log_softmax(self.fc2(output), dim=1)
        return (output, x1, x2, x3)

class Net3(nn.Module) :
    """
    Implements a trainable CNN with three convolutional layer and two fully connected layers
    #### PARAMETERS ####
    dropout_rate: dropout regularization rate
    n_filters: number of filters for to create
    filter_size: size of filters in filter_size x filter_size
    fc_units: numer of units in the first fully-connected layer
    """
    def __init__(self, dropout_rate=0.2, n_filters=10, filter_size=5, fc_units=50) :
        super(Net3, self).__init__()
        self.dropout_rate=dropout_rate
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.fc_units = fc_units
        self.conv1  = nn.Conv2d(1, self.n_filters, kernel_size=self.filter_size, padding=2)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters*2, kernel_size=self.filter_size, padding=2)
        self.conv3 = nn.Conv2d(self.n_filters*2, self.n_filters*4, kernel_size=self.filter_size, padding=2)
        self.fc1 = nn.Linear(self.n_filters*4*3*3, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, 10)

    def forward(self, x) :
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(F.max_pool2d(self.conv3(x2), 2))
        x4 = x3.view(-1, self.n_filters*4*3*3)
        x4 = F.relu(self.fc1(x4))
        output = F.dropout(x4, p=self.dropout_rate, training=self.training)
        output = F.log_softmax(self.fc2(output), dim=1)
        return (output, x1, x2, x3, x4)

class Net4(nn.Module) :
    """
    Implements a trainable CNN with four convolutional layers and two fully connected layers
    #### PARAMETERS ####
    dropout_rate: dropout regularization rate
    n_filters: number of filters for to create
    filter_size: size of filters in filter_size x filter_size
    fc_units: numer of units in the first fully-connected layer
    """
    def __init__(self, dropout_rate=0.2, n_filters=10, filter_size=5, fc_units=50) :
        super(Net4, self).__init__()
        self.dropout_rate=dropout_rate
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.fc_units = fc_units
        self.conv1  = nn.Conv2d(1, self.n_filters, kernel_size=self.filter_size, padding=2)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters*2, kernel_size=self.filter_size, padding=2)
        self.conv3 = nn.Conv2d(self.n_filters*2, self.n_filters*4, kernel_size=self.filter_size, padding=2)
        self.conv4= nn.Conv2d(self.n_filters*4, self.n_filters*8, kernel_size=self.filter_size, padding=2)
        self.fc1 = nn.Linear(self.n_filters*8*1*1, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, 10)

    def forward(self, x) :
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(F.max_pool2d(self.conv3(x2), 2))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x5 = x4.view(-1, self.n_filters*8*1*1)
        x5 = F.relu(self.fc1(x5))
        output = F.dropout(x5, p=self.dropout_rate, training=self.training)
        output = F.log_softmax(self.fc2(output), dim=1)
        return (output, x1, x2, x3, x4, x5)

def train(model, optimizer, train_loader, use_cuda, device, epoch, layers=1):
    """
    Train the given n (between 1 and 4) layer CNN for 1 epoch and print performance
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = helper.rotate(data, cuda=use_cuda)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if layers == 1:
            output, h_conv1, h_fc1 = model(data)
        elif layers == 2:
            output, h_conv1, h_conv2, h_fc1 = model(data)
        elif layers == 3:
            output, h_conv1, h_conv2, h_conv3, h_fc1 = model(data)
        elif layers == 4:
            output, h_conv1, h_conv2, h_conv3, h_conv4, h_fc1 = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test(model, optimizer, test_loader, use_cuda, device, epoch, layers=1):
    model.eval()
    test_loss = 0.
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data = helper.rotate(data, cuda=use_cuda)
            data, target = data.to(device), target.to(device)
            if layers == 1:
                output, h_conv1, h_fc1 = model(data)
            elif layers == 2:
                output, h_conv1, h_conv2, h_fc1 = model(data)
            elif layers == 3:
                output, h_conv1, h_conv2, h_conv3, h_fc1 = model(data)
            elif layers == 4:
                output, h_conv1, h_conv2, h_conv3, h_conv4, h_fc1 = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return (test_loss, correct / len(test_loader.dataset))
