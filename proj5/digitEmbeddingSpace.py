# Your name here and a short header
# Yueyang Wu

# import statements
from torchvision import datasets, transforms

import utils
from utils import MyNetwork
from utils import SubModel

import numpy as np
import sys
import csv
import torch
from torch.utils.data import DataLoader
import torchvision


def main(argv):
    # load greek dataset
    image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/greek'
    greek = datasets.ImageFolder(image_dir,
                                 transform=transforms.Compose([transforms.Resize((28, 28)),
                                                               transforms.Grayscale(),
                                                               transforms.functional.invert,
                                                               transforms.ToTensor()]))

    # write dataset to csv files
    utils.write_to_csv(greek)




    return


if __name__ == "__main__":
    main(sys.argv)