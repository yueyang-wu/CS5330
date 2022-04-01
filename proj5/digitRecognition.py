# Your name here and a short header
# Yueyang Wu


# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt


# class definitions
class DataSet():
    def __init__(self):
        pass


class MyNetwork(nn.Module):
    def __init__(self):
        pass


# computes a forward pass for the network
# methods need a summary comment
def forward(self, x):
    return x


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

    return


if __name__ == "__main__":
    main(sys.argv)
