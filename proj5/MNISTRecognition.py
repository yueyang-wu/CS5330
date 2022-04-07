# Your name here and a short header
# Yueyang Wu

# import statements
import utils

import ssl
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context


# main function (yes, it needs a comment too)
def main(argv):
    # make the network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # load test and training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=utils.BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=utils.BATCH_SIZE_TEST, shuffle=True)

    # plot the first 6 example digits
    utils.plot_images(train_loader, 2, 3)

    # initialize the network and the optimizer
    network = utils.MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=utils.LEARNING_RATE,
                          momentum=utils.MOMENTUM)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(utils.N_EPOCHS + 1)]

    # run the training
    utils.test(network, test_loader, test_losses)
    for epoch in range(1, utils.N_EPOCHS + 1):
        utils.train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        utils.test(network, test_loader, test_losses)

    # plot training curve
    utils.plot_curve(train_counter, train_losses, test_counter, test_losses)

    # save the model
    torch.save(network, 'model.pth')

    return


if __name__ == "__main__":
    main(sys.argv)
