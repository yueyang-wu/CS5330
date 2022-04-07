# Your name here and a short header
# Yueyang Wu


# import statements
import sys
import torch
from torch import nn
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

# define hyper-parameters
n_epochs = 5
batch_size_train = 64
batch_size = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10

torch.set_printoptions(precision=2)


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


def plot_prediction(data, output, row, col):
    fig = plt.figure()
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main(argv):
    # load the model
    model = torch.load('model.pth')

    # load test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    first_ten_data = []
    first_ten_label = []
    count = 0
    for data, target in test_loader:
        if count < 10:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_ten_data.append(squeeze_data)
            # print(data.shape)
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - output: {output}')
                print(f'{count + 1} - index of the max output value: {output.argmax().item()}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - prediction label: {label}')
                first_ten_label.append(label)
                count += 1
                # plt.imshow(np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0)), cmap='gray', interpolation='none')
                # plt.show()

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(first_ten_data[i], cmap='gray', interpolation='none')
        plt.title('Prediction: {}'.format(first_ten_label[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
