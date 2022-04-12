"""
Yueyang Wu

This file contains some pre-defined hyper-parameters,
model definition, and helper functions definition
"""

# import statements
import csv
from collections import Counter

import cv2
import torch
from numpy import linalg
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# define hyper-parameters
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10


# model definition

# A deep network with the following layers:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
# A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
# A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
class MyNetwork(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        x = x.view(-1, 320)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 50 nodes
        x = self.fc2(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, 1)  # apply log_softmax()


# A truncated deep network contains the first two convolution layers of MyNetwork:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
class SubModel(MyNetwork):
    # initialize the network layers, inherit from MyNetwork
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(
            F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x


# A deep network inherits MyNetwork, which terminates at the Dense layer with 50 outputs:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
# A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
class DigitEmbeddingModel(MyNetwork):
    # initialize the network layers, inherit from MyNetwork
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x


# useful functions with a comment for each function

'''
The function plots images

@:parameter data: the images to be plotted
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot
'''
def plot_images(data, row, col):
    examples = enumerate(data)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


'''
The function trains the model and save the model and optimizer

@:parameter epoch: number of epochs of the training process
@:parameter model: the model to be trained
@:parameter optimizer: the optimizer used when training
@:parameter train_loader: the training data
@:parameter train_losses: array to record the train losses
@:parameter train_counter: array to record the train counter
'''
def train(epoch, model, optimizer, train_loader, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


'''
The function tests the model and print the accuracy information

@:parameter model: the model to be tested
@:parameter test_loader: the test data
@:parameter test_losses: array to record test losses
'''
def test(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


'''
The function plots curves of the training loses and testing losses

@:parameter train_counter: array of train counter
@:parameter train_losses: array of train losses
@:parameter test_counter: array of test counter
@:parameter test_losses: array of test losses
'''
def plot_curve(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


'''
The function apply model on dataset and get the first 10 data and the labels
@:parameter data: the testing data
@:parameter model: the model used

@:return first_ten_data: array contains the first 10 data
@:return first_ten_label: array contains the label of the first 10 data
'''
def first_ten_output(data, model):
    first_ten_data = []
    first_ten_label = []

    count = 0
    for data, target in data:
        if count < 10:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_ten_data.append(squeeze_data)
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - output: {output}')
                print(f'{count + 1} - index of the max output value: {output.argmax().item()}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - prediction label: {label}')
                first_ten_label.append(label)
                count += 1

    return first_ten_data, first_ten_label


'''
The function plots the image and their prediction values

@:parameter data_set: the image to be plotted
@:parameter label_set: the labels of the dataset
@:parameter total: total number of data to be plotted
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot
'''
def plot_prediction(data_set, label_set, total, row, col):
    for i in range(total):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data_set[i], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(label_set[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


'''
The function plots some filters

@:parameter conv: the convolutation layer from a model which contains the filters to be plotted
@:parameter total: total number of filters to be plotted
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot

@:return filters:; array of all the filters plotted
'''
def plot_filters(conv, total, row, col):
    filters = []
    with torch.no_grad():
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            curr_filter = conv.weight[i, 0]
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
    return filters


'''
The function plots filters and filtered images

@:parameter filters: the filters to be plotted
@:parameter image: the image to be filtered
@:parameter n: the total number of filters
@:parameter total: total number of images in the plot
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot
'''
def plot_filtered_images(filters, image, n, total, row, col):
    with torch.no_grad():
        items = []
        for i in range(n):
            items.append(filters[i])
            filtered_image = cv2.filter2D(np.array(image), ddepth=-1, kernel=np.array(filters[i]))
            items.append(filtered_image)
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            plt.imshow(items[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()


'''
The function take some labeled images, and write their intensity values and category to two csv files

@:parameter data: the labeled images
@:parameter filename1: the csv file path to write the intensity values
@:parameter filename2: the csv file path to write the categories
'''
def write_to_csv(data, filename1, filename2):
    intensity_values = open(filename1, 'w')
    intensity_values_writer = csv.writer(intensity_values)
    category = open(filename2, 'w')
    category_writer = csv.writer(category)

    intensity_values_header = []
    for i in range(784):
        intensity_values_header.append(str(i))
    category_header = ['label']
    intensity_values_writer.writerow(intensity_values_header)
    category_writer.writerow(category_header)

    for image, label in data:
        row = []
        image_np = image.numpy()
        for x in np.nditer(image_np):
            row.append(str(x))

        intensity_values_writer.writerow(row)
        category_writer.writerow([str(label)])


'''
The function applies a model to the given image(read the intensity values from csv file) to get element vectors

@:parameter model: the model to be applied
@:parameter greek_dir: the csv file contains the intensity values

@:return outputs: array contains the element vectors
'''
def project_greek(model, greek_dir):
    outputs = []
    greek_dir = greek_dir
    with open(greek_dir) as greek_obj:
        heading = next(greek_obj)
        reader_obj = csv.reader(greek_obj)
        for row in reader_obj:  # row -> list of str
            row_float = []
            for r in row:
                row_float.append(float(r))
            image_tensor = torch.Tensor(row_float)
            resize_tensor = image_tensor.view(1, 28, 28)
            output = model(resize_tensor)
            a = output.detach().numpy()[0]
            outputs.append(a / linalg.norm(a))
    return outputs


'''
The function computes the sum-squared distance of two arrays

@:parameter arr1: the first array
@:parameter arr2: the second array

@:return ans: the ssd of the two arrays
'''
def ssd(arr1, arr2):
    ans = 0
    for i in range(len(arr1)):
        ans += (arr1[i] - arr2[i]) ** 2
    return ans


'''
The function computes the ssd values between on array and some other arrays

@:parameter arr: the array to be computed
@:parameter arrays: an array of all the other arrays

@:return ans: an array contains all the ssd values between arr and the other arrays
'''
def all_ssd(arr, arrays):
    ans = []
    for i in range(len(arrays)):
        ans.append(ssd(arr, arrays[i]))
    return ans


'''
The function plots the ssd values computeed by all_ssd() as a dot plot

@:parameter embedding: the output values got after applying models to images
@:parameter outputs: the output values got after applying models to images
@:parameter examples: the index of sample alpha, beta, and gamma from outputs
@:parameter num_index: the length of x axis
'''
def print_plot_ssd(embedding, outputs, examples, num_index):
    index = []
    for i in range(num_index):
        index.append(i + 1)
    for e in examples:
        values = all_ssd(outputs[e], embedding)
        print(values)
        plt.scatter(index, values)
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.show()


'''
The function classifies the image category using KNN classifier

@:parameter distances: array of all distances calculated by all_ssd()
@:parameter category: array of category of the training data
@:parameter k: the K value in KNN classifier

@return: the category name
'''
def knn(distances, category, k):
    sorted_idx = np.argsort(distances)
    codes = []
    for i in range(k):
        name_code = category[sorted_idx[i]]
        codes.append(name_code)
    count_dict = Counter(codes)
    max_name_code = max(count_dict, key=count_dict.get)
    return get_class_name(max_name_code)


'''
The function gets the category name of greek letters given the name code

@:parameter code: the given code

@:return: the corresponding name in string
'''
def get_class_name(code):
    name_dict = {1: 'alpha', 2: 'beta', 3: 'gamma', 4: 'delta', 5: 'epsilon', 6: 'zeta', 7: 'eta',
                 8: 'theta', 9: 'iota', 10: 'kappa', 11: 'lambda', 12: 'mu', 13: 'nu', 14: 'xi',
                 15: 'omicron', 16: 'pi', 17: 'rho', 18: 'sigma', 19: 'tau', 20: 'upsilon',
                 21: 'phi', 22: 'chi', 23: 'psi', 24: 'omega'}
    return name_dict.get(code + 1)
