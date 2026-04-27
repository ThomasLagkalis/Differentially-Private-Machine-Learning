from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_reconstruction(x_rec, epoch, i, title="Reconstruction", cmap="gray"):
    """
    Displays a flattened MNIST reconstruction vector as an image.

    Args:
        x_rec: torch.Tensor or numpy array of shape (784,) or (1, 784)
        title: plot title
    """
    if hasattr(x_rec, "detach"):   # torch tensor
        x_rec = x_rec.detach().cpu().numpy()

    x_rec = np.squeeze(x_rec)      # remove batch dim if present
    img = x_rec.reshape(32, 32)

    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.title(title)
    plt.savefig(f'./results/reconstructions/rec{epoch+1}_{i+1}.png')

def moving_avg(x, window=10):
    x = np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")

def load_mnist_lenet(batch_size=64, n_train=None, n_test=None):
    """
    Loads MNIST resized to 32x32 and returns PyTorch DataLoaders
    ready for LeNet.

    Args:
        batch_size: batch size for training
        n_train: optional subset size for training
        n_test: optional subset size for test

    Returns:
        train_loader, test_loader
    """

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),   # -> (1, 32, 32)
    ])

    train_set = datasets.MNIST(
        root="./data/mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.MNIST(
        root="./data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    # Optional subsampling
    if n_train is not None:
        idx = torch.randperm(len(train_set))[:n_train]
        train_set = Subset(train_set, idx)

    if n_test is not None:
        idx = torch.randperm(len(test_set))[:n_test]
        test_set = Subset(test_set, idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def prepare_mnist(n):
    '''
    Loads and prepares the MNIST dataset. Takes as input the number of training points n and return X, y, means.
    '''

    # Define transform: convert to tensor [0,1] and flatten to 784-dim vector
    transform = transforms.Compose([
        transforms.ToTensor(),                      # -> (1, 28, 28), values in [0,1]
        transforms.Lambda(lambda x: x.view(-1))     # -> (784,)
    ])
    

    # Load the training dataset
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

    # Load the test dataset
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    # Pick a smaller subset of training data (e.g. 5000 samples)
    subset_size = n
    indices = torch.randperm(len(train_dataset))[:subset_size]  # random subset
    train_dataset = Subset(train_dataset, indices)

    # Keep test dataset small as well if needed (e.g. 1000 samples)
    # Probably irrelevant.
    #test_subset_size = n/10
    #test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
    #test_dataset = Subset(test_dataset, test_indices)

    # Convert subset to arrays X, y
    X = []
    y = []
    for i in range(len(train_dataset)):
        data, label = train_dataset[i]
        X.append(data.numpy())   # convert tensor -> numpy
        y.append(label)

    X = np.stack(X)   # shape (n, 784)

    y = np.array(y)   # shape (n,)

    return X, y

def prepare_data_loader(X, y, batch_size=1):
    '''
    Prepares the data loader and returns it.
    Input: - X: the data.
           - y: the labels of the data.
           - batch_size. Default is 1.
    Returns: a DataLoader object ready for training and one for testing and the total number of training data.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_samples = len(train_loader) * train_loader.batch_size

    return train_loader, test_loader, total_samples


