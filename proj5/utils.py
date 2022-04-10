# Your name here and a short header
# Yueyang Wu

# import statements
import csv

import cv2
import torch
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
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)


class SubModel(MyNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(
            F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        return x


class DigitEmbeddingModel(MyNetwork):
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


def train(epoch, model, optimizer, train_loader, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


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


def plot_curve(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


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


def plot_prediction(data_set, label_set, total, row, col):
    for i in range(total):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data_set[i], cmap='gray', interpolation='none')
        plt.title('Prediction: {}'.format(label_set[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_ten_filters(model):
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
    return filters


def plot_twenty_filters(model):
    filters = []
    with torch.no_grad():
        for i in range(20):
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            curr_filter = model.conv2.weight[i, 0]
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


# name1 intensity values, name2 category
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
            outputs.append(output.detach().numpy()[0])
    return outputs


def ssd(arr1, arr2):
    ans = 0
    for i in range(len(arr1)):
        ans += (arr1[i] - arr2[i]) ** 2
    return ans


def all_ssd(arr, arrays):
    ans = []
    for i in range(len(arrays)):
        ans.append(ssd(arr, arrays[i]))
    return ans


# examples: the index of sample alpha, beta, and gamma from outputs
def print_plot_ssd(outputs, examples, num_index):
    index = []
    for i in range(num_index):
        index.append(i + 1)
    for e in examples:
        values = all_ssd(outputs[e], outputs)
        print(values)
        plt.plot(index, values)
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.show()
