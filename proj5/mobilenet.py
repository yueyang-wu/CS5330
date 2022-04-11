import ssl
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn


# allow using unverified SSL due to some configuration issue
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utils

ssl._create_default_https_context = ssl._create_unverified_context


class MobilenetSubModel(nn.Module):
    def __init__(self):
        super(MobilenetSubModel, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.conv1 = list(model.features)[0][0] # only keep the first conv layer

    def forward(self, x):
        return self.conv1(x)


def main(argv):
    # initialize a model, set to eval mode
    mobilenet = MobilenetSubModel()
    mobilenet.eval()

    # load data
    custom_image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/dog'
    custom_greek = datasets.ImageFolder(custom_image_dir,
                                        transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.1307,), (0.3081,))]))

    #  get the first image
    first_image, first_label = next(iter(custom_greek))
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    filters = utils.plot_filters(mobilenet.conv1, 32, 4, 8)

    # plot the first images filtered by the 10 filters from layer 1
    utils.plot_filtered_images(filters, squeezed_image, 32, 64, 8, 8)


if __name__ == "__main__":
    main(sys.argv)
