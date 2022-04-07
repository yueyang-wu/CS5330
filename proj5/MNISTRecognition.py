# Your name here and a short header
# Yueyang Wu

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import matplotlib.pyplot as plt

# allow using unverified SSL due to some configuration issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# define hyper-parameters
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


# class definitions
# class MyNetwork(nn.Module):
#     def __init__(self):
#         super(MyNetwork, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
#         self.max_pool = nn.MaxPool2d((2, 2))
#         # nn.ReLU()
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
#         self.dropout = nn.Dropout2d()
#         # nn.MaxPool2d((2, 2))
#         # nn.ReLU()
#         # nn.Flatten()
#         self.fc1 = nn.Linear(320, 50)
#         # nn.ReLU()
#         self.fc2 = nn.Linear(50, 10)
#
#     # computes a forward pass for the network
#     # methods need a summary comment
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.dropout(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, 0)
#         return output

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, 1)


# useful functions with a comment for each function
def plot_images(data, row, col):
    examples = enumerate(data)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train(epoch, network, optimizer, train_loader, train_losses, train_counter):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def plot_curve(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code
    # make the network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # load test and training data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    # plot the first 6 example digits
    plot_images(train_loader, 2, 3)

    # initialize the network and the optimizer
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # run the training
    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        test(network, test_loader, test_losses)

    # plot training curve
    plot_curve(train_counter, train_losses, test_counter, test_losses)

    # save the model
    torch.save(network, 'model.pth')
    # torch.save(network.state_dict(), 'model_weights.pth')

    return


if __name__ == "__main__":
    main(sys.argv)
