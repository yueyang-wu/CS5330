'''
Yueyang Wu

CS5330 Project 5 Extension 3
'''

# import statements
import csv
import sys
import torch
import numpy as np
from torchvision import datasets, transforms

import utils


'''
Load a digit embedding model, which inherits MyNetwork and terminates at the Dense layer with 50 outputs
Load a full set of Greek letters training dataset, which contains examples of all the 24 Greek letters (10 for each)
Write the intensity values and category of the images to two csv files
Apply the model to the dataset, and get the element vectors

Load a full set of Greek letters testing dataset, which contains examples of all the 24 Greek letters (4 for each)
For each image, apply the model on it and get teh element vector of the image
Apply KNN classifier on the element vector of the image and the element vectors of the training dataset, 
then get the prediction value of the testing image
Plot the testing image and their predicted labels
'''
def main(argv):
    # build a new model
    digit_embedding_model = utils.DigitEmbeddingModel()
    digit_embedding_model.load_state_dict(torch.load('model_state_dict.pth'))
    digit_embedding_model.eval()

    # load greek dataset
    image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/full_greek/train_letters_images'
    full_greek = datasets.ImageFolder(image_dir,
                                      transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                    transforms.Grayscale(),
                                                                    transforms.functional.invert,
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))]))

    # write dataset to csv files
    utils.write_to_csv(full_greek, 'full_intensity_values.csv', 'full_category.csv')

    # project greek digits into the embedding space
    full_greek_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/full_intensity_values.csv'
    outputs = utils.project_greek(digit_embedding_model, full_greek_dir)  # 240 array with len(array) = 50

    # construct category[], corresponding to outputs
    category_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/full_category.csv'
    category = []  # len(category) should be 240
    with open(category_dir) as category_obj:
        heading = next(category_obj)
        reader_obj = csv.reader(category_obj)
        for row in reader_obj:
            category.append(int(row[0]))

    # load custom greek symbols
    test_full_image_dir = '/Users/yueyangwu/Desktop/CS5330/hw/proj5/full_greek/test_letters_images'
    test_full_greek = datasets.ImageFolder(test_full_image_dir,
                                           transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                         transforms.Grayscale(),
                                                                         transforms.functional.invert,
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))]))

    images = []
    predictions = []
    with torch.no_grad():
        for image, label in test_full_greek:
            output_feature = digit_embedding_model(image)[0]
            squeezed_image = np.transpose(torch.squeeze(image, 1).detach().numpy(), (1, 2, 0))
            images.append(squeezed_image)

            distance = utils.all_ssd(output_feature, outputs)
            pred = utils.knn(distance, category, 5)
            predictions.append(pred)

    utils.plot_prediction(images, predictions, 96, 8, 12)

    return


if __name__ == "__main__":
    main(sys.argv)
