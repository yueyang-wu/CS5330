'''
Yueyang Wu

CS5330 Project 5 Task 3
'''

# import statements
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import utils


'''
Load the given Greek dataset, which contains examples of alpha, beta, and gamma(3 for each)
Write the intensity values and category of the images to two csv files
Load a digit embedding model, which inherits MyNetwork and terminates at the Dense layer with 50 outputs
Load the MNIST training dataset, apply the model to the first image and print out the shape of the result

Apply the model to the Greek dataset and get the element vectors
Compute the ssd of some example images and plot the result

Load a custom Greek digit dataset
Apply the model to the custom Greek dataset and get the element vectors
Compute the ssd of the custom images and plot the result
'''
def main(argv):
    # load greek dataset
    image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/greek'
    greek = datasets.ImageFolder(image_dir,
                                 transform=transforms.Compose([transforms.Resize((28, 28)),
                                                               transforms.Grayscale(),
                                                               transforms.functional.invert,
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.1307,), (0.3081,))]))

    # write dataset to csv files
    utils.write_to_csv(greek, 'intensity_values.csv', 'category.csv')

    # build a new model
    digit_embedding_model = utils.DigitEmbeddingModel()
    digit_embedding_model.load_state_dict(torch.load('model_state_dict.pth'))
    digit_embedding_model.eval()

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
    output = digit_embedding_model(first_image)
    print(output.shape)

    # project greek digits into the embedding space
    greek_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/intensity_values.csv'
    outputs = utils.project_greek(digit_embedding_model, greek_dir)

    # compute distance in the embedding space and plot the distances
    utils.print_plot_ssd(outputs, outputs, [0, 9, 18], 27)

    # load custom greek symbols
    custom_image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/custom_greek2'
    custom_greek = datasets.ImageFolder(custom_image_dir,
                                        transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                      transforms.Grayscale(),
                                                                      transforms.functional.invert,
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.1307,), (0.3081,))]))

    # write dataset to csv files
    utils.write_to_csv(custom_greek, 'custom_intensity_values.csv', 'custom_category.csv')
    custom_greek_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/custom_intensity_values.csv'
    # apply the model and display the SSD of outputs
    custom_outputs = utils.project_greek(digit_embedding_model, custom_greek_dir)
    utils.print_plot_ssd(outputs, custom_outputs, [0, 1, 2], 27)
    return


if __name__ == "__main__":
    main(sys.argv)
