# Your name here and a short header
# Yueyang Wu

# import statements
import numpy as np
from matplotlib import pyplot as plt

import utils
from utils import MyNetwork

import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import cv2


def main(argv):
    # load and print the model
    model = torch.load('model.pth')
    print(model)

    # for conv1 layer, print the filter weights and the shape
    # plot the 10 filters
    filters = []
    with torch.no_grad():
        for i in range(10):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            curr_filter = model.conv1.weight[i, 0]
            filters.append(curr_filter)
            print(f'filter {i + 1}')
            print(curr_filter)
            print(curr_filter.shape)
            print('\n')
            plt.imshow(curr_filter)
            plt.title(f'Filter {i + 1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()

    # apply the 10 filters to the first training example image
    # load training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))
    #  get the first image
    first_image, first_label = next(iter(train_loader))
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))
    plt.imshow(squeezed_image)
    plt.show()

    with torch.no_grad():
        items = []
        for i in range(10):
            items.append(filters[i])
            filtered_image = cv2.filter2D(np.array(squeezed_image), ddepth=-1, kernel=np.array(filters[i]))
            items.append(filtered_image)
        for i in range(20):
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            plt.imshow(items[i])
            # plt.imshow(cv2.filter2D(np.array(first_image), ddepth=-1, kernel=np.array(filters[i])))
            plt.xticks([])
            plt.yticks([])
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
