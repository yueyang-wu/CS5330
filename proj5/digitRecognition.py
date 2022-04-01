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
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import matplotlib.pyplot as plt


# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(5, 5)),
            nn.Dropout(0.5),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 20, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.layer_stack(x)
        output = F.log_softmax(x)
        return output


# useful functions with a comment for each function
def plot_images(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 2
    for i in range(1, cols * rows + 1):
        sample_idx = i - 1
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f'Ground Truth: {label}')
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


def train_network(arguments):
    return


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # main function code
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False

    # load training and test data
    training_data = datasets.MNIST(root='data',
                                   train=True,
                                   download=True,
                                   transform=ToTensor()
                                   )
    test_data = datasets.MNIST(root='data',
                               train=False,
                               download=True,
                               transform=ToTensor()
                               )

    # plot the first 6 example digits
    plot_images(training_data)

    # prepare data for training with DataLoader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # transform
    ds = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

    # get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # create an instance of MyNetwork and move it to the device and print its structure
    model = MyNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f'Predicted class: {y_pred}')
    return


if __name__ == "__main__":
    main(sys.argv)
