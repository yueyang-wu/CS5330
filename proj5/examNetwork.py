# Your name here and a short header
# Yueyang Wu

# import statements
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import mnist

import utils
from utils import MyNetwork
from utils import SubModel

import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2


def main(argv):
    # load and print the model
    model = torch.load('model.pth')
    print(model)

    # for conv1 layer, print the filter weights and the shape
    # plot the 10 filters
    filters = utils.plot_ten_filters(model)

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

    utils.plot_filtered_images(filters, squeezed_image, 10, 20, 5, 4)

    sub_model = SubModel()
    sub_model.load_state_dict(torch.load('model_state_dict.pth'))
    sub_model.eval()

    truncated_filters = utils.plot_twenty_filters(sub_model)
    utils.plot_filtered_images(truncated_filters, squeezed_image, 20, 40, 5, 8)


if __name__ == "__main__":
    main(sys.argv)
