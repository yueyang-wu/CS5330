# Your name here and a short header
# Yueyang Wu

# import statements
from matplotlib import pyplot as plt

import utils
from utils import MyNetwork

import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms


def main(argv):
    # load and print the model
    model = torch.load('model.pth')
    print(model)

    # for conv1 layer, print the filter weights and the shape
    # plot the 10 filters
    with torch.no_grad():
        for i in range(10):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            curr_filter = model.conv1.weight[i, 0]
            print(f'filter {i + 1}')
            print(curr_filter)
            print(curr_filter.shape)
            print('\n')
            plt.imshow(curr_filter)
            plt.title(f'Filter {i + 1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
