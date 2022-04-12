'''
Yueyang Wu

CS5330 Project 5 Task 2
'''

# import statements
import utils
from utils import MyNetwork
from utils import SubModel

import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import torchvision


'''
Load the model trained by MNISTRecognition.py
Plot the 10 filters in the first convolution layer of the model

Load the MNIST training data
Apply the 10 filters to the first image in the training dataset
Plot the 10 filters and 10 filtered images

Load a truncated model from the previous model
Apply the 20 filters from the second convolution layer of this model to the first image from the training dataset
Plot the 20 filters and the 20 filtered images
'''
def main(argv):
    # load and print the model
    model = torch.load('model.pth')
    print(model)

    # for conv1 layer, print the filter weights and the shape
    # plot the 10 filters
    filters = utils.plot_filters(model.conv1, 10, 3, 4)

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

    # plot the first images filtered by the 10 filters from layer 1
    utils.plot_filtered_images(filters, squeezed_image, 10, 20, 5, 4)

    # load a sub model object and load the state dictionary
    sub_model = SubModel()
    sub_model.load_state_dict(torch.load('model_state_dict.pth'))
    sub_model.eval()

    # apply the sub model on the first image and plot the filtered images
    truncated_filters = utils.plot_filters(sub_model.conv2, 20, 5, 4)
    utils.plot_filtered_images(truncated_filters, squeezed_image, 20, 40, 5, 8)


if __name__ == "__main__":
    main(sys.argv)
