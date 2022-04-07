# Your name here and a short header
# Yueyang Wu

# import statements
import utils
from utils import MyNetwork

import sys
import torch
from torch.utils.data import DataLoader
import torchvision

# set output precision to 2
torch.set_printoptions(precision=2)


def main(argv):
    # load the model
    model = torch.load('model.pth')

    # load test data
    test_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    # get the label of the first ten images and print out the outputs
    first_ten_data, first_ten_label = utils.first_ten_output(test_loader, model)

    # plot the predictions for the first ten images
    utils.plot_prediction(first_ten_data, first_ten_label, 3, 3)

    return


if __name__ == "__main__":
    main(sys.argv)
