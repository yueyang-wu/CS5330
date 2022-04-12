'''
Yueyang Wu

CS5330 Project 5 Extension 2
'''

import ssl
import sys

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

import utils

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context


'''
A deep network modified from the pre-trained mobilenet from PyTorch
Only contains the first convolution layer of the original mobilenet network
'''
class MobilenetSubModel(nn.Module):
    # initialize the model
    def __init__(self):
        super(MobilenetSubModel, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.conv1 = list(model.features)[0][0] # only keep the first conv layer

    # compute a forward pass of the first convolution layer
    def forward(self, x):
        return self.conv1(x)


'''
Initialize the modified mobilenet network
Load an image of a dog
Apply the first 32 filters of the first convolution layer to the dog image
Plot the 32 filters and the 32 filtered dog images
'''
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
